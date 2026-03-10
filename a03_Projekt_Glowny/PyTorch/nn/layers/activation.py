import logging
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from common_utils import class_autologger
from nn import LayerProtocol, Tensor, Tensor2D, validate_shape

Mtx: TypeAlias = np.ndarray


def apply_relu_forward_logic(x: Tensor) -> tuple[Tensor, Tensor]:
    validate_shape(x, (-1, x.ndim))

    mask = x > 0
    return np.where(mask, x, 0), mask


def apply_relu_backward_logic(grad: Tensor, mask: Tensor) -> Tensor:
    new_grad = grad.copy()
    new_grad[~mask] = 0
    return new_grad


def apply_sigmoid_forward_logic(x: Tensor) -> Tensor:
    validate_shape(x, (-1, x.ndim))

    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def apply_sigmoid_backward_logic(grad: Tensor, last_output: Tensor) -> Tensor:
    return last_output * (1 - last_output) * grad


def apply_softmax_forward_logic(x: Tensor2D) -> Tensor2D:
    validate_shape(x, (-1, 2))
    x = np.array([])
    return x.astype(Mtx)


def apply_softmax_backward_logic(grad: Tensor2D, last_output: Tensor2D) -> Tensor2D:
    validate_shape(grad, (-1, 2))
    x = np.array([])
    return x.astype(Mtx)


@class_autologger
@dataclass
class ReLULayer:
    _input_mask: Tensor = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=bool)
    )

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: Tensor) -> Tensor:
        result, self._input_mask = apply_relu_forward_logic(x)
        return result

    def backward(self, grad: Tensor) -> Tensor:
        return apply_relu_backward_logic(grad, self._input_mask)


@class_autologger
@dataclass
class SigmoidLayer:
    _last_output: Tensor = field(
        init=False, repr=False, default_factory=lambda: np.array([])
    )

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: Tensor) -> Tensor:
        self._last_output = apply_sigmoid_forward_logic(x)
        return self._last_output

    def backward(self, grad: Tensor) -> Tensor:
        return apply_sigmoid_backward_logic(grad, self._last_output)


@class_autologger
@dataclass
class SoftmaxLayer(LayerProtocol):
    _last_output: Tensor2D = field(
        init=False, repr=False, default_factory=lambda: np.array([[]])
    )

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: Tensor2D) -> Tensor2D:
        self._last_output = apply_softmax_forward_logic(x)
        return self._last_output

    def backward(self, grad: Tensor2D) -> Tensor2D:
        return apply_softmax_backward_logic(grad, self._last_output)
