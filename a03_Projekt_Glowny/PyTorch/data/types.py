from typing import Annotated, Any, Dict, List, Optional, TypeAlias, Union

import numpy as np
from nn import Tensor2D

Label: TypeAlias = int
LabelsMtx: TypeAlias = np.ndarray
BatchData: TypeAlias = List[np.ndarray]

JsonDict: TypeAlias = Dict[str, Any]
JsonList: TypeAlias = List[JsonDict]
JsonData: TypeAlias = Union[JsonDict, JsonList]
ImageBytes: TypeAlias = bytes
FilePath: TypeAlias = str
DataResult: TypeAlias = Union[FilePath, ImageBytes, JsonData]

OptLabel: TypeAlias = Optional[Label]
OptLabelsMtx: TypeAlias = Optional[LabelsMtx]
OptBatchData: TypeAlias = Optional[BatchData]
OptFilePath: TypeAlias = Optional[FilePath]
OptJsonData: TypeAlias = Optional[JsonData]

Sample: TypeAlias = tuple[np.ndarray, Label]
DatasetBatch: TypeAlias = tuple[BatchData, LabelsMtx]

ImageGray: TypeAlias = Annotated[np.ndarray, "H, W"]
ImageRGB: TypeAlias = Annotated[np.ndarray, "H, W, 3"]

RawImage: TypeAlias = np.ndarray
ProcessedImage: TypeAlias = Tensor2D


def is_grayscale(data: np.ndarray) -> bool:
    return data.ndim == 2


def is_color(data: np.ndarray) -> bool:
    return data.ndim == 3 and data.shape[-1] == 3
