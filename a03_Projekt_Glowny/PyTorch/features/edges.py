from typing import Protocol, TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class EdgeDetectorProtocol(Protocol):
    def apply(self, matrix: Mtx) -> Mtx: ...


class Sobel:
    def apply(self, matrix: Mtx) -> Mtx:
        return matrix


class Prewitt:
    def apply(self, matrix: Mtx) -> Mtx:
        return matrix
