from typing import Protocol

import numpy as np

Mtx = np.ndarray
MtxList = list[np.ndarray]


class BatchProcessingProtocol(Protocol):
    def create_batches(
        self, data: MtxList, batch_size: int, shuffle: bool = True
    ) -> list[MtxList]: ...

    def process_batch(self, paths: list[str]) -> MtxList: ...


class BatchProcessing:
    def create_batches(
        self, data: MtxList, batch_size: int, shuffle: bool
    ) -> list[MtxList]:
        ...

    def process_batch(self, paths: list[str]) -> MtxList:
        ...
