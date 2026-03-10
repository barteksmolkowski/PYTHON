import logging
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from common_utils import class_autologger
from nn import T2D, LayerProtocol, T, validate_shape

Mtx: TypeAlias = np.ndarray


def relu_fwd(x: T) -> tuple[T, T]:
    validate_shape(x, (-1, x.ndim))

    mask = x > 0
    return np.where(mask, x, 0), mask


def relu_bwd(grad: T, mask: T) -> T:
    new_grad = grad.copy()
    new_grad[~mask] = 0
    return new_grad


def sigmoid_fwd(x: T) -> T:
    validate_shape(x, (-1, x.ndim))

    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def sigmoid_bwd(grad: T, last_output: T) -> T:
    return last_output * (1 - last_output) * grad


def softmax_fwd(x: T2D) -> T2D:
    validate_shape(x, (-1, 2))
    x = np.array([])
    return x.astype(Mtx)


def softmax_bwd(grad: T2D, last_output: T2D) -> T2D:
    validate_shape(grad, (-1, 2))
    x = np.array([])
    return x.astype(Mtx)


@class_autologger
@dataclass
class ReLU:
    _input_mask: T = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=bool)
    )

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: T) -> T:
        result, self._input_mask = relu_fwd(x)
        return result

    def backward(self, grad: T) -> T:
        return relu_bwd(grad, self._input_mask)


@class_autologger
@dataclass
class Sigmoid:
    _last_output: T = field(
        init=False, repr=False, default_factory=lambda: np.array([])
    )

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: T) -> T:
        self._last_output = sigmoid_fwd(x)
        return self._last_output

    def backward(self, grad: T) -> T:
        return sigmoid_bwd(grad, self._last_output)


@class_autologger
@dataclass
class Softmax(LayerProtocol):
    _last_output: T2D = field(
        init=False, repr=False, default_factory=lambda: np.array([[]])
    )

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: T2D) -> T2D:
        self._last_output = softmax_fwd(x)
        return self._last_output

    def backward(self, grad: T2D) -> T2D:
        return softmax_bwd(grad, self._last_output)
