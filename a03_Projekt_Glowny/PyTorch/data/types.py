from typing import Any, Dict, List, Optional, TypeAlias, Union

import numpy as np

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
