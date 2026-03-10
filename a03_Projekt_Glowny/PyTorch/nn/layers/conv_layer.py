import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from common_utils import class_autologger
from nn import Mtx, Tensor4D, validate_shape


def apply_conv2d_forward_logic(
    x: Tensor4D, kernels: Mtx, biases: Mtx, stride: int
) -> Tensor4D:
    validate_shape(x, (-1, 4))
    return np.zeros_like(x)


def apply_conv2d_backward_logic(
    grad: Tensor4D, x_cache: Tensor4D, kernels: Mtx, stride: int
) -> Tuple[Tensor4D, Mtx, Mtx]:
    validate_shape(grad, (-1, 4))
    validate_shape(x_cache, (-1, 4))

    d_x = np.zeros_like(x_cache)
    d_kernels = np.zeros_like(kernels)
    d_biases = np.zeros(kernels.shape[0])

    return d_x, d_kernels, d_biases


@class_autologger
@dataclass
class Conv2DLayer:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1

    kernels: Mtx = field(default_factory=lambda: np.array([]))
    biases: Mtx = field(default_factory=lambda: np.array([]))

    _input_cache: Tensor4D = field(
        init=False, repr=False, default_factory=lambda: np.array([])
    )

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.kernels.size == 0:
            self.kernels = np.random.randn(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
            )

        if self.biases.size == 0:
            self.biases = np.zeros(self.out_channels)

    def forward(self, x: Tensor4D) -> Tensor4D:
        self._input_cache = x
        return apply_conv2d_forward_logic(x, self.kernels, self.biases, self.stride)

    def backward(self, grad: Tensor4D) -> Tensor4D:
        d_x, d_kernels, d_biases = apply_conv2d_backward_logic(
            grad, self._input_cache, self.kernels, self.stride
        )
        return d_x
