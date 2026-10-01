"""in_parallel and map_concurrently: concurrent fetches that lose no result and no error."""

import threading

import pytest

from finance_mcp.data.concurrency import in_background, in_parallel, map_concurrently


def test_in_parallel_keeps_the_second_error_when_both_fail() -> None:
    def first() -> None:
        raise ValueError("first failed")

    def second() -> None:
        raise KeyError("second failed")

    with pytest.raises(ValueError, match="first failed") as excinfo:
        in_parallel(first, second)
    assert any("second failed" in note for note in excinfo.value.__notes__)


def test_in_parallel_raises_the_second_error_when_only_it_fails() -> None:
    def second() -> None:
        raise KeyError("second failed")

    with pytest.raises(KeyError, match="second failed"):
        in_parallel(lambda: 1, second)


def test_in_parallel_runs_both_at_once() -> None:
    # The barrier only opens when both are in flight together.
    gate = threading.Barrier(2, timeout=10)

    def after_the_gate(value: str) -> str:
        gate.wait()
        return value

    assert in_parallel(lambda: after_the_gate("a"), lambda: after_the_gate("b")) == ("a", "b")


def test_map_concurrently_keeps_input_order() -> None:
    assert map_concurrently(["a", "b", "c"], str.upper, max_workers=2) == ["A", "B", "C"]


def test_map_concurrently_of_nothing_is_empty() -> None:
    assert map_concurrently([], str.upper, max_workers=4) == []


def test_in_background_runs_while_the_block_does() -> None:
    started = threading.Event()

    def fetch() -> str:
        started.set()
        return "bills"

    with in_background(fetch) as future:
        assert started.wait(timeout=10)
        assert future.result() == "bills"
