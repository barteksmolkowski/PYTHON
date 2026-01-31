import types

# from . import decorators as _dec
from .augmentation import DataAugmentation
from .conversion import ImageToMatrixConverter
from .convolution import ConvolutionActions
from .decorators import *
from .geometry import ImageGeometry
from .grayscale import GrayScaleProcessing
from .io_image import ImageHandler
from .normalization import Normalization
from .pipeline import ImageDataPreprocessing, TransformPipeline
from .pooling import Pooling
from .thresholding import Thresholding

base_attrs = dir(types.ModuleType("base"))
base_attrs.extend(
    [
        "__annotations__",
        "__builtins__",
        "__file__",
        "__cached__",
        "types",
        "__path__",
        "__loader__",
        "__spec__",
    ]
)

imported_names = [
    n
    for n in locals()
    if n not in base_attrs and n != "base_attrs" and (n.startswith("__") or "_" in n)
]

__all__ = imported_names

if "base_attrs" in locals():
    del base_attrs
if "imported_names" in locals():
    del imported_names
