from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    ParamSpec,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
from jaxtyping import Float, Int, Shaped, UInt8

P = ParamSpec("P")
T_Var = TypeVar("T_Var")
T_Obj = TypeVar("T_Obj")

if TYPE_CHECKING:
    from jaxtyping import Bool, Float, Int, Shaped, UInt8, jaxtyped
    from typeguard import typechecked as typechecker
else:
    try:
        from jaxtyping import Bool, Float, Int, Shaped, UInt8
        from typeguard import typechecked as typechecker

        def jaxtyped(typechecker=None):
            return lambda x: x
    except ImportError:

        class _Fallback:
            def __getitem__(self, _):
                return np.ndarray

        Bool = Float = Int = Shaped = UInt8 = _Fallback()

        def jaxtyped(typechecker=None):
            return lambda x: x

        def typechecker(x):
            return x


T: TypeAlias = Annotated[Float[np.ndarray, "..."], "Generic"]
T1D: TypeAlias = Annotated[npt.NDArray[np.float32], Float[np.ndarray, "Features"]]
T2D: TypeAlias = Annotated[npt.NDArray[np.float32], Float[np.ndarray, "Batch Features"]]
T3D: TypeAlias = Annotated[
    npt.NDArray[np.float32], Float[np.ndarray, "Batch Seq Hidden"]
]
T4D: TypeAlias = Annotated[
    npt.NDArray[np.float32], Float[np.ndarray, "Batch Channels Height Width"]
]
M4D: TypeAlias = Annotated[
    npt.NDArray[np.bool_], Bool[np.ndarray, "Batch Channels Height Width"]
]
ImageGray: TypeAlias = Annotated[npt.NDArray[Any], Shaped[np.ndarray, "Height Width"]]
Padded: TypeAlias = Annotated[
    npt.NDArray[np.float32], Shaped[np.ndarray, "H_pad W_pad"]
]
PaddedImage: TypeAlias = Padded
ImageRGB: TypeAlias = Annotated[npt.NDArray[Any], Shaped[np.ndarray, "Height Width 3"]]
RawImage: TypeAlias = Annotated[
    npt.NDArray[np.uint8], UInt8[np.ndarray, "Height Width Channels"]
]
LabelsMtx: TypeAlias = Annotated[npt.NDArray[np.int64], Int[np.ndarray, "Batch"]]

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
ClassType: TypeAlias = Type[T_Obj]
ProcessorFunc: TypeAlias = Callable[[ImageGray], T]
MetricsDict: TypeAlias = Dict[str, Union[float, int, str]]
ResultWithMetrics: TypeAlias = Tuple[T, MetricsDict]
Strings: TypeAlias = Union[str, List[str]]
FuncDec: TypeAlias = Union[Callable[P, T_Var], List[Callable[P, T_Var]]]
ClsDec: TypeAlias = Callable[[ClassType], ClassType]


__all__ = [
    "M4D",
    "T1D",
    "T2D",
    "T3D",
    "T4D",
    "BatchData",
    "ClassType",
    "ClsDec",
    "DataResult",
    "DatasetBatch",
    "FilePath",
    "FuncDec",
    "ImageBytes",
    "ImageGray",
    "ImageRGB",
    "JsonData",
    "JsonDict",
    "JsonList",
    "Label",
    "LabelsMtx",
    "MetricsDict",
    "Padded",
    "PaddedImage",
    "ProcessorFunc",
    "RawImage",
    "ResultWithMetrics",
    "Sample",
    "Shape",
    "Strings",
    "T",
    "T_Obj",
    "jaxtyped",
    "typechecker",
]
