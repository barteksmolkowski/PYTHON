from typing import Protocol

import numpy as np

Mtx = np.ndarray
MtxList = list[np.ndarray]


class EdgeDetectorProtocol(Protocol):
    def apply(self, matrix: Mtx) -> Mtx: ...


class Sobel:
    def apply(self, matrix: Mtx) -> Mtx:
        return matrix


class Prewitt:
    def apply(self, matrix: Mtx) -> Mtx:
        return matrix
