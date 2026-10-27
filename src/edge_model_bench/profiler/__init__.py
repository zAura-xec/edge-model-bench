"""Profiling utilities."""

from typing import Any

__all__ = ["HbrMultiModelProfiler"]


def __getattr__(name: str) -> Any:
    if name == "HbrMultiModelProfiler":
        from .hbr import HbrMultiModelProfiler

        return HbrMultiModelProfiler
    raise AttributeError(name)
