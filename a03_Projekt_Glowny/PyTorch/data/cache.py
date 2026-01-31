from typing import Optional, Protocol, runtime_checkable

import numpy as np

Mtx = np.ndarray
MtxList = list[np.ndarray]


class CacheManagerProtocol(Protocol):
    def cache(self, data: MtxList) -> bool: ...

    def load(self) -> Optional[MtxList]: ...


class CacheManager:
    def cache(self, data: MtxList) -> bool:
        """
        data - lista macierzy do zapisania w pamięci podręcznej

        return bool (czy zapis się udał)
        """
        return False

    def load(self) -> Optional[MtxList]:
        """
        wczytuje dane z pamięci podręcznej

        return MtxList lub None jeśli cache jest pusty
        """
        return None
