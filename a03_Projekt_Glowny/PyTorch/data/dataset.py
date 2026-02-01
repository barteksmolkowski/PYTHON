from typing import Protocol, Union

import numpy as np

Mtx = np.ndarray
MtxList = list[np.ndarray]


class DatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Union[Mtx, MtxList]: ...


class Dataset:
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Union[Mtx, MtxList]:
        ...
