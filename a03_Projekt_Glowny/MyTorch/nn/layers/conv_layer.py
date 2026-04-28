from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from interfaces import LayerABC

from MyTorch import T4D, T


def conv2d_fwd(x: T4D, kernels: T4D, biases: T, stride: int = 1) -> T4D:
    return np.array([], dtype=float)


def conv2d_bwd(
    grad: T4D, x_cache: T4D, kernels: T4D, stride: int = 1
) -> Tuple[T4D, T4D, T]:
    d_x: T4D = np.array([], dtype=float)
    d_kernels: T4D = np.array([], dtype=float)
    d_biases: T = np.array([], dtype=float)

    return d_x, d_kernels, d_biases


@dataclass(frozen=True, slots=True)
class Conv2D(LayerABC):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1

    kernels: T4D = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=float)
    )
    biases: T = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=float)
    )
    _input_cache: T4D = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=float)
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.kernels.size == 0:
            kernels_val = np.random.randn(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
            ).astype(float)
            object.__setattr__(self, "kernels", kernels_val)

        if self.biases.size == 0:
            biases_val = np.zeros(self.out_channels, dtype=float)
            object.__setattr__(self, "biases", biases_val)

    def forward(self, x: T4D) -> T4D:
        object.__setattr__(self, "_input_cache", x)
        return conv2d_fwd(x, self.kernels, self.biases, self.stride)

    def backward(self, grad: T4D) -> T4D:
        return conv2d_bwd(grad, self._input_cache, self.kernels, self.stride)[0]
