import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from common_utils import class_autologger
from nn import Mtx, T, validate_shape


def dropout_fwd(x: T, probability: float, training: bool = True) -> Tuple[T, T]:
    validate_shape(x, (-1, x.ndim))

    if not training or probability <= 0:
        return x, np.ones_like(x, dtype=bool)

    mask = np.random.rand(*x.shape) > probability
    scale = 1.0 / (1.0 - probability)
    result = x * mask * scale

    return result, mask


def dropout_bwd(grad: Mtx, mask: Mtx, probability: float) -> Mtx:
    x = np.array([]).astype(Mtx)
    return x


@class_autologger
@dataclass
class Dropout:
    probability: float = 0.5
    training: bool = True

    _mask: T = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=bool)
    )

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if not (0 <= self.probability < 1):
            self.logger.error(f"Invalid dropout probability: {self.probability}")
            raise ValueError("Probability must be in range [0, 1)")

    def forward(self, x: T) -> T:
        result, self._mask = dropout_fwd(x, self.probability, self.training)
        return result

    def backward(self, grad: T) -> T:
        return dropout_bwd(grad, self._mask, self.probability)
