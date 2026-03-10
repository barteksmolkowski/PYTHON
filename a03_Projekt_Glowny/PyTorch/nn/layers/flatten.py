import logging
from dataclasses import dataclass, field

import numpy as np
from common_utils import class_autologger
from nn import T2D, T4D, Shape, validate_shape


def flatten_fwd(x: T4D) -> tuple[T2D, Shape]:
    validate_shape(x, (-1, 4))

    original_shape = x.shape
    result = x.reshape(x.shape[0], -1)

    return result, original_shape


def flatten_bwd(grad: T2D, original_shape: Shape) -> T4D:
    validate_shape(grad, (-1, 2))

    return grad.reshape(original_shape)


@class_autologger
@dataclass
class FlattenLayer:
    _original_shape: Shape = field(init=False, repr=False, default_factory=tuple)

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: T4D) -> T2D:
        result, self._original_shape = flatten_fwd(x)
        return result

    def backward(self, grad: T2D) -> T4D:
        return flatten_bwd(grad, self._original_shape)
