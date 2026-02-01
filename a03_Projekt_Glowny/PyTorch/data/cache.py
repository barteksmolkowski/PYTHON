from typing import Optional, Protocol, TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class CacheManagerProtocol(Protocol):
    def cache(self, data: MtxList) -> bool: ...

    def load(self) -> Optional[MtxList]: ...


class CacheManager:
    def cache(self, data: MtxList) -> bool: ...

    def load(self) -> Optional[MtxList]: ...
