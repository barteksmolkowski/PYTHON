from typing import Protocol, Union, runtime_checkable

import numpy as np

Mtx = np.ndarray
MtxList = list[np.ndarray]


class DatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Union[Mtx, MtxList]: ...


class Dataset:
    def __len__(self) -> int:
        """
        zwraca całkowitą liczbę próbek w zbiorze danych
        """
        return 0

    def __getitem__(self, index: int) -> Union[Mtx, MtxList]:
        """
        pobiera pojedynczą macierz lub listę macierzy dla danego indeksu
        """
        return np.array([])
