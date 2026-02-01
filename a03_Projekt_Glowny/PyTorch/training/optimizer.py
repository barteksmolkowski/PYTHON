from typing import Protocol, TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray


class OptimizerProtocol(Protocol):
    def update(self, weights: Mtx, grad: Mtx) -> Mtx: ...


class SGD:
    def __init__(self) -> None:
        pass

    def update(self, weights: Mtx, grad: Mtx) -> Mtx:
        pass


class Adam:
    def __init__(self) -> None:
        pass

    def update(self, weights: Mtx, grad: Mtx) -> Mtx:
        pass
