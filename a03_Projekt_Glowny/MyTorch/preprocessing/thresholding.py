from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from MyTorch import T2D, ImageGray

from .base import ThresholdingABC
from .decorators import parameter_complement


def generate_gaussian_kernel(size: int) -> T2D:
    v = size - np.abs(np.arange(-size + 1, size))
    return np.outer(v, v).astype(np.float32)


def adaptive_threshold_logic(
    matrix: ImageGray, block_size: int, c: int, kernel: T2D
) -> ImageGray:
    pad_size: int = block_size // 2
    padded = np.pad(matrix, pad_width=pad_size, mode="reflect")
    windows = sliding_window_view(padded, (block_size, block_size))

    kernel_sum: float = float(np.sum(kernel))
    local_means = np.sum(windows * kernel, axis=(2, 3)) / kernel_sum

    result = np.where(matrix > (local_means - c), 255.0, 0.0).astype(np.float32)
    return result


@dataclass(frozen=True, slots=True)
class Thresholding(ThresholdingABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    @parameter_complement
    def adaptive_threshold(
        self,
        matrix: ImageGray,
        block_size: int = 5,
        c: int = 2,
        auto_params: bool = True,
    ) -> ImageGray:
        pad_size: int = block_size // 2
        kernel: T2D = generate_gaussian_kernel(pad_size + 1)
        return adaptive_threshold_logic(
            matrix=matrix, block_size=block_size, c=c, kernel=kernel
        )
