from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from interfaces import LayerABC

from MyTorch import M4D, T4D


def dropout_fwd(x: T4D, probability: float, training: bool = True) -> Tuple[T4D, M4D]:
    if not training or probability <= 0.0:
        return x, np.ones(x.shape, dtype=bool)

    mask = np.random.random(x.shape) > probability

    scale = 1.0 / (1.0 - probability)
    result = (x * mask * scale).astype(np.float32)

    return result, mask


def dropout_bwd(grad: T4D, mask: M4D, probability: float) -> T4D:
    return np.array([], dtype=np.float32).reshape(0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class Dropout(LayerABC):
    probability: float = 0.5
    training: bool = True

    _mask: M4D = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=bool)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0 <= self.probability < 1):
            raise ValueError(f"Probability {self.probability} must be in range [0, 1)")

    def forward(self, x: T4D) -> T4D:
        result, mask = dropout_fwd(x, self.probability, self.training)
        object.__setattr__(self, "_mask", mask)
        return result

    def backward(self, grad: T4D) -> T4D:
        return dropout_bwd(grad, self._mask, self.probability)
