"""Structural backend contract; implementations need not inherit a base class."""

from typing import Any, Protocol


class Runner(Protocol):
    latencies: list[float]  # One positive duration per timed call, in seconds.

    def __call__(self, inputs: Any) -> Any: ...

    def close(self) -> None: ...
