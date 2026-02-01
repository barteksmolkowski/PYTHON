from typing import Optional, Protocol

import numpy as np

Mtx = np.ndarray
MtxList = list[np.ndarray]


class CacheManagerProtocol(Protocol):
    def cache(self, data: MtxList) -> bool: ...

    def load(self) -> Optional[MtxList]: ...


class CacheManager:
    def cache(self, data: MtxList) -> bool:
        ...

    def load(self) -> Optional[MtxList]:
        ...
