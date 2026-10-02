"""Run independent blocking fetches at the same time.

Providers are synchronous, so independent lookups (a batch of quotes, an asset and its
benchmark) run on worker threads instead of one after another.
"""

from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager


def map_concurrently[T](items: list[str], fetch: Callable[[str], T], max_workers: int) -> list[T]:
    """Run ``fetch`` over ``items`` in parallel, returning results in input order."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(fetch, items))


def in_parallel[A, B](first: Callable[[], A], second: Callable[[], B]) -> tuple[A, B]:
    """Run two independent blocking fetches at once; either one's exception propagates.

    If both fail, the first error is raised with the second attached as a note, so
    neither is lost.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first)
        second_future = pool.submit(second)
        try:
            first_result = first_future.result()
        except Exception as exc:
            second_error = second_future.exception()
            if second_error is not None:
                exc.add_note(f"The fetch run alongside it also failed: {second_error!r}")
            raise
        return first_result, second_future.result()


@contextmanager
def in_background[T](fetch: Callable[[], T]) -> Iterator[Future[T]]:
    """Start ``fetch`` on a worker thread for the duration of the ``with`` block.

    For a result several other fetches each need once they have their own data; leaving
    the block waits for it to finish.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        yield pool.submit(fetch)
