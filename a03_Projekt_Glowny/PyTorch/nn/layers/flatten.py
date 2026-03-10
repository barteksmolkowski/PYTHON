import logging
from dataclasses import dataclass, field

import numpy as np
from common_utils import class_autologger
from nn import Shape, Tensor2D, Tensor4D, validate_shape


def apply_flatten_forward_logic(x: Tensor4D) -> tuple[Tensor2D, Shape]:
    validate_shape(x, (-1, 4))

    original_shape = x.shape
    result = x.reshape(x.shape[0], -1)

    return result, original_shape


def apply_flatten_backward_logic(grad: Tensor2D, original_shape: Shape) -> Tensor4D:
    validate_shape(grad, (-1, 2))

    return grad.reshape(original_shape)


@class_autologger
@dataclass
class FlattenLayer:
    _original_shape: Shape = field(init=False, repr=False, default_factory=tuple)

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: Tensor4D) -> Tensor2D:
        result, self._original_shape = apply_flatten_forward_logic(x)
        return result

    def backward(self, grad: Tensor2D) -> Tensor4D:
        return apply_flatten_backward_logic(grad, self._original_shape)
