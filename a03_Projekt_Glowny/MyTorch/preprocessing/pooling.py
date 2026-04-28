from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from MyTorch import ImageGray, Shape

from .base import PoolingABC


def max_pool_logic(
    matrix: ImageGray,
    kernel_size: Shape,
    stride: int,
    pad_width: int,
    pad_values: int,
) -> ImageGray:
    m_np: ImageGray = matrix.astype(np.float32)
    if pad_width > 0:
        m_np = np.pad(
            m_np,
            pad_width=pad_width,
            mode="constant",
            constant_values=float(pad_values),
        ).astype(np.float32)

    windows = sliding_window_view(m_np, window_shape=kernel_size)
    view = windows[::stride, ::stride]
    result: ImageGray = np.max(view, axis=(2, 3)).astype(np.float32)
    return result


@dataclass(frozen=True, slots=True)
class Pooling(PoolingABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    def max_pool(
        self,
        matrix: ImageGray,
        kernel_size: Shape = (2, 2),
        stride: int = 0,
        pad_width: int = 0,
        pad_values: int = 0,
    ) -> ImageGray:
        actual_stride: int = stride if stride > 0 else int(kernel_size[0])
        return max_pool_logic(
            matrix=matrix,
            kernel_size=kernel_size,
            stride=actual_stride,
            pad_width=pad_width,
            pad_values=pad_values,
        )
