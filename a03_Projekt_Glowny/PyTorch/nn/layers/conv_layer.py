import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from common_utils import class_autologger
from nn import T4D, Mtx, validate_shape


def conv2d_fwd(x: T4D, kernels: Mtx, biases: Mtx, stride: int) -> T4D:
    validate_shape(x, (-1, 4))
    return np.zeros_like(x)


def conv2d_bwd(
    grad: T4D, x_cache: T4D, kernels: Mtx, stride: int
) -> Tuple[T4D, Mtx, Mtx]:
    validate_shape(grad, (-1, 4))
    validate_shape(x_cache, (-1, 4))

    d_x = np.zeros_like(x_cache)
    d_kernels = np.zeros_like(kernels)
    d_biases = np.zeros(kernels.shape[0])

    return d_x, d_kernels, d_biases


@class_autologger
@dataclass
class Conv2D:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1

    kernels: Mtx = field(init=False, repr=False, default_factory=lambda: np.array([]))
    biases: Mtx = field(init=False, repr=False, default_factory=lambda: np.array([]))

    _input_cache: T4D = field(
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

    def forward(self, x: T4D) -> T4D:
        self._input_cache = x
        return conv2d_fwd(x, self.kernels, self.biases, self.stride)

    def backward(self, grad: T4D) -> T4D:
        d_x, d_kernels, d_biases = conv2d_bwd(
            grad, self._input_cache, self.kernels, self.stride
        )
        return d_x
