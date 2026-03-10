import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from common_utils import class_autologger
from nn import Mtx, Tensor2D, validate_shape


def apply_linear_forward_logic(x: Tensor2D, weights: Mtx, bias: Mtx) -> Tensor2D:
    validate_shape(x, (-1, 2))

    if x.shape[1] != weights.shape[0]:
        raise ValueError(
            f"Input features {x.shape[1]} != weight features {weights.shape[0]}"
        )
    return np.dot(x, weights) + bias


def apply_linear_backward_logic(
    grad: Tensor2D, x_cache: Tensor2D, weights: Mtx
) -> Tuple[Tensor2D, Mtx, Mtx]:
    validate_shape(grad, (-1, 2))

    grad_input = np.dot(grad, weights.T)
    grad_weights = np.dot(x_cache.T, grad)
    grad_bias = np.sum(grad, axis=0)
    return grad_input, grad_weights, grad_bias


@class_autologger
@dataclass
class LinearLayer:
    in_features: int
    out_features: int

    weights: Mtx = field(default_factory=lambda: np.array([]))
    bias: Mtx = field(default_factory=lambda: np.array([]))

    _x_cache: Tensor2D = field(
        init=False, repr=False, default_factory=lambda: np.array([[]])
    )

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.weights.size == 0:
            limit = np.sqrt(6 / (self.in_features + self.out_features))
            self.weights = np.random.uniform(
                -limit, limit, (self.in_features, self.out_features)
            )

        if self.bias.size == 0:
            self.bias = np.zeros(self.out_features)

    def forward(self, x: Tensor2D) -> Tensor2D:
        self._x_cache = x
        return apply_linear_forward_logic(x, self.weights, self.bias)

    def backward(self, grad: Tensor2D) -> Tensor2D:
        grad_input, grad_w, grad_b = apply_linear_backward_logic(
            grad, self._x_cache, self.weights
        )
        return grad_input
