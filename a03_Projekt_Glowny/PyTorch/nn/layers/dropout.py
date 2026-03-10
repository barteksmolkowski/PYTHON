import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from common_utils import class_autologger
from nn import Mtx, Tensor, validate_shape


def apply_dropout_forward_logic(
    x: Tensor, probability: float, training: bool = True
) -> Tuple[Tensor, Tensor]:
    validate_shape(x, (-1, x.ndim))

    if not training or probability <= 0:
        return x, np.ones_like(x, dtype=bool)

    mask = np.random.rand(*x.shape) > probability
    scale = 1.0 / (1.0 - probability)
    result = x * mask * scale

    return result, mask


def apply_dropout_backward_logic(grad: Mtx, mask: Mtx, probability: float) -> Mtx:
    x = np.array([]).astype(Mtx)
    return x


@class_autologger
@dataclass
class DropoutLayer:
    probability: float = 0.5
    training: bool = True

    _mask: Tensor = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=bool)
    )

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if not (0 <= self.probability < 1):
            self.logger.error(f"Invalid dropout probability: {self.probability}")
            raise ValueError("Probability must be in range [0, 1)")

    def forward(self, x: Tensor) -> Tensor:
        result, self._mask = apply_dropout_forward_logic(
            x, self.probability, self.training
        )
        return result

    def backward(self, grad: Tensor) -> Tensor:
        return apply_dropout_backward_logic(grad, self._mask, self.probability)
