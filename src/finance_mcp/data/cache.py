"""A thread-safe TTL cache with an LRU bound on the number of entries."""

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, cast

CacheKey = tuple[str, ...]


class TTLCache:
    """Per-key TTL, least-recently-used eviction past ``max_entries``.

    Bookkeeping is guarded by a lock because callers fetch concurrently. Fetches run
    OUTSIDE the lock: two threads racing on one uncached key just fetch it twice.
    """

    def __init__(self, time_fn: Callable[[], float], max_entries: int) -> None:
        self._now = time_fn
        self._max_entries = max_entries
        # key -> (stored_at, ttl, value), in LRU order (oldest use first). The ttl is kept
        # per entry so the purge pass can judge expiry without knowing who wrote it.
        self._entries: OrderedDict[CacheKey, tuple[float, float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_fetch[T](
        self,
        key: CacheKey,
        ttl: float,
        fetch: Callable[[], T],
        cacheable: Callable[[T], bool] | None = None,
    ) -> T:
        """Return ``fetch()``, reusing a live entry and storing the result under ``key``.

        ``cacheable`` is consulted after the fetch to decide whether the value is worth
        keeping; ``None`` means always keep it. It bounds an entry by size, which the
        entry-count LRU cannot do.
        """
        now = self._now()
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                if now - hit[0] < ttl:
                    self._entries.move_to_end(key)
                    # Values are Any, but each key prefix always stores the same type.
                    return cast(T, hit[2])
                del self._entries[key]
        value = fetch()
        if cacheable is not None and not cacheable(value):
            return value
        stored_at = self._now()  # read AFTER the fetch: a slow fetch must not age its entry
        with self._lock:
            self._purge_expired(stored_at)
            self._entries[key] = (stored_at, ttl, value)
            # Assigning a key that is still present (a concurrent fetch of the same key
            # got there first) leaves it in its old position, so order it explicitly.
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def _purge_expired(self, now: float) -> None:
        """Drop every entry past its own TTL, so stale keys cannot squat on the bound."""
        for key in [k for k, (stored, ttl, _) in self._entries.items() if now - stored >= ttl]:
            del self._entries[key]
