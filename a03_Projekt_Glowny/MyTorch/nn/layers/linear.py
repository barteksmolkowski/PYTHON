from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from interfaces import LayerABC

from MyTorch import T2D


def linear_fwd(x: T2D, weights: T2D, bias: T2D) -> T2D:
    return (x @ weights + bias).astype(np.float32)


def linear_bwd(grad: T2D, x_cache: T2D, weights: T2D) -> Tuple[T2D, T2D, T2D]:
    grad_input = (grad @ weights.T).astype(np.float32)
    grad_weights = (x_cache.T @ grad).astype(np.float32)
    grad_bias = np.sum(grad, axis=0, keepdims=True).astype(np.float32)
    return grad_input, grad_weights, grad_bias


@dataclass(frozen=True, slots=True)
class Linear(LayerABC):
    in_features: int
    out_features: int
    weights: T2D = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=np.float32)
    )
    bias: T2D = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=np.float32)
    )
    _input_cache: T2D = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=np.float32)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.weights.size == 0:
            limit = np.sqrt(2.0 / self.in_features)
            w_val = np.random.randn(self.in_features, self.out_features) * limit
            object.__setattr__(self, "weights", w_val.astype(np.float32))
        if self.bias.size == 0:
            b_val = np.zeros((1, self.out_features), dtype=np.float32)
            object.__setattr__(self, "bias", b_val)

    def forward(self, x: T2D) -> T2D:
        object.__setattr__(self, "_input_cache", x)
        return linear_fwd(x, self.weights, self.bias)

    def backward(self, grad: T2D) -> T2D:
        grad_input, grad_w, grad_b = linear_bwd(grad, self._input_cache, self.weights)
        return grad_input
