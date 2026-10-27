"""Import only the selected converter; vendor SDKs are optional."""

from importlib import import_module

_EXPORTS = {
    "MtnnConvertConfig": "mtnn",
    "MtnnPipeline": "mtnn",
    "OnnxStaticQuantConfig": "static_onnx",
    "OnnxStaticQuantizer": "static_onnx",
    "StaticOnnxMtnnPipeline": "static_mtnn",
    "HbrConvertConfig": "hbr",
    "HbrOnnxOpsetDowngrader": "hbr",
    "HbrPipeline": "hbr",
    "HbrQuantConfig": "hbr",
    "DlcToBin": "qnn",
    "OnnxToDlc": "qnn",
    "QnnConfig": "qnn",
    "QnnPipeline": "qnn",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
