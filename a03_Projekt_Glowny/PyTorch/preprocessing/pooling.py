from typing import Optional, Protocol

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

Mtx = np.ndarray
Kernel = tuple[int, int]


class PoolingProtocol(Protocol):
    def max_pool(
        self,
        matrix: Mtx,
        kernel_size: Kernel = (2, 2),
        stride: Optional[int] = None,
        pad_width: int = 0,
        pad_values: int = 0,
    ) -> Mtx: ...


class Pooling:
    def max_pool(
        self,
        matrix: Mtx,
        kernel_size: Kernel = (2, 2),
        stride: Optional[int] = None,
        pad_width: int = 0,
        pad_values: int = 0,
    ) -> Mtx:
        M = np.asanyarray(matrix, dtype=np.float32)
        if stride == None:
            stride = kernel_size[0]

        if pad_width > 0:
            M = np.pad(
                M, pad_width=pad_width, mode="constant", constant_values=pad_values
            )

        windows = sliding_window_view(M, window_shape=kernel_size)

        view = windows[::stride, ::stride]

        return np.max(view, axis=(2, 3)).astype(np.uint8)
