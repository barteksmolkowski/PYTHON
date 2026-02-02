from typing import Protocol, TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray


class HOGProtocol(Protocol):
    def extract(self, matrix: Mtx) -> list[float]: ...


class HOG:
    def extract(self, matrix: Mtx) -> list[float]:
        return [0.0]
