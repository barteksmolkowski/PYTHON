from typing import Optional, Protocol, TypeAlias

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from common_utils import class_autologger

Mtx: TypeAlias = np.ndarray
Kernel: TypeAlias = tuple[int, int]


class PoolingProtocol(Protocol):
    def max_pool(
        self,
        matrix: Mtx,
        kernel_size: Kernel = (2, 2),
        stride: Optional[int] = None,
        pad_width: int = 0,
        pad_values: int = 0,
    ) -> Mtx: ...


@class_autologger
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
            self.logger.debug(
                f"[max_pool] Stride not provided. Defaulting to kernel_size[0]: {stride}"
            )

        if pad_width > 0:
            self.logger.debug(
                f"[max_pool] Applying padding: width={pad_width}, value={pad_values}. Original shape: {M.shape}"
            )
            M = np.pad(
                M, pad_width=pad_width, mode="constant", constant_values=pad_values
            )
            self.logger.debug(f"[max_pool] Matrix shape after padding: {M.shape}")

        windows = sliding_window_view(M, window_shape=kernel_size)
        view = windows[::stride, ::stride]

        self.logger.debug(
            f"[max_pool] Creating pooling view with kernel={kernel_size} and stride={stride}. View grid shape: {view.shape[:2]}"
        )

        return np.max(view, axis=(2, 3)).astype(np.uint8)
