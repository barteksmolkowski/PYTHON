from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

import numpy as np
from jaxtyping import Float, Int, Shaped, UInt8

if TYPE_CHECKING:
    from jaxtyping import Float, Int, Shaped, UInt8, jaxtyped
    from typeguard import typechecked as typechecker
else:
    try:
        from jaxtyping import Float, Int, Shaped, UInt8
        from typeguard import typechecked as typechecker

        def jaxtyped(typechecker=None):
            return lambda x: x
    except ImportError:

        class _Fallback:
            def __getitem__(self, _):
                return np.ndarray

        Float = Int = Shaped = UInt8 = _Fallback()

        def jaxtyped(typechecker=None):
            return lambda x: x

        def typechecker(x):
            return x


T: TypeAlias = Annotated[Float[np.ndarray, "..."], "Generic Array"]
T2D: TypeAlias = Annotated[Float[np.ndarray, "Batch Features"], "2D Array"]
T3D: TypeAlias = Annotated[Float[np.ndarray, "Batch Seq Hidden"], "3D Array"]
T4D: TypeAlias = Annotated[Float[np.ndarray, "Batch Channels Height Width"], "4D Array"]

ImageGray: TypeAlias = Annotated[Shaped[np.ndarray, "Height Width"], "Grayscale Image"]
ImageRGB: TypeAlias = Annotated[Shaped[np.ndarray, "Height Width 3"], "RGB Image"]
RawImage: TypeAlias = Annotated[UInt8[np.ndarray, "Height Width Channels"], "Raw Image"]
LabelsMtx: TypeAlias = Annotated[Int[np.ndarray, "Batch"], "Labels Matrix"]

Label: TypeAlias = int
BatchData: TypeAlias = List[T]
Sample: TypeAlias = Tuple[T, Label]
DatasetBatch: TypeAlias = Tuple[BatchData, LabelsMtx]

JsonDict: TypeAlias = Dict[str, Any]
JsonList: TypeAlias = List[JsonDict]
JsonData: TypeAlias = Union[JsonDict, JsonList]
FilePath: TypeAlias = str
ImageBytes: TypeAlias = bytes
DataResult: TypeAlias = Union[FilePath, ImageBytes, JsonData]
Shape: TypeAlias = Tuple[int, ...]

T_Obj = TypeVar("T_Obj")
ClassType: TypeAlias = Type[T_Obj]
ProcessorFunc: TypeAlias = Callable[[ImageGray], T]
MetricsDict: TypeAlias = Dict[str, Union[float, int, str]]
ResultWithMetrics: TypeAlias = Tuple[T, MetricsDict]

__all__ = [
    "T2D",
    "T3D",
    "T4D",
    "BatchData",
    "ClassType",
    "DataResult",
    "DatasetBatch",
    "FilePath",
    "ImageBytes",
    "ImageGray",
    "ImageRGB",
    "JsonData",
    "JsonDict",
    "JsonList",
    "Label",
    "LabelsMtx",
    "MetricsDict",
    "ProcessorFunc",
    "RawImage",
    "ResultWithMetrics",
    "Sample",
    "Shape",
    "T",
    "T_Obj",
    "jaxtyped",
    "typechecker",
]
