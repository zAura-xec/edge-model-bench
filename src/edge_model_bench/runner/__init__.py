from typing import Any

__all__ = ["HbrRunner", "MtnnRunner", "OnnxRunner", "QnnRunner"]


def __getattr__(name: str) -> Any:
    if name == "HbrRunner":
        from .hbr import HbrRunner

        return HbrRunner
    if name == "MtnnRunner":
        from .mtnn import MtnnRunner

        return MtnnRunner
    if name == "OnnxRunner":
        from .onnx import OnnxRunner

        return OnnxRunner
    if name == "QnnRunner":
        from .qnn import QnnRunner

        return QnnRunner
    raise AttributeError(name)
