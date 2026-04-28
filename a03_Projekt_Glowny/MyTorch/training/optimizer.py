from dataclasses import dataclass

import numpy as np
from interfaces import OptimizerABC

from MyTorch import T2D


@dataclass(frozen=True, slots=True)
class SGD(OptimizerABC):
    learning_rate: float = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()

    def update(self, weights: T2D, grad: T2D) -> T2D:
        return np.array([[]], dtype=float)


@dataclass(frozen=True, slots=True)
class Adam(OptimizerABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    def update(self, weights: T2D, grad: T2D) -> T2D:
        return np.array([[]], dtype=float)
