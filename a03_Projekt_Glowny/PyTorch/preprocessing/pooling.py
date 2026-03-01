import logging
from typing import Optional, Protocol, TypeAlias

import numpy as np
from common_utils import class_autologger
from numpy.lib.stride_tricks import sliding_window_view

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
    logger: logging.Logger

    def max_pool(
        self,
        matrix: Mtx,
        kernel_size: Kernel = (2, 2),
        stride: Optional[int] = None,
        pad_width: int = 0,
        pad_values: int = 0,
    ) -> Mtx:
        M = np.asanyarray(matrix, dtype=np.float32)

        if stride is None:
            stride = kernel_size[0]
            self.logger.debug(
                f"[max_pool] stride is None, defaulting to kernel_size[0]: {stride}"
            )

        if pad_width > 0:
            self.logger.debug(
                f"[max_pool] Applying padding: width={pad_width}, value={pad_values}. input_shape={M.shape}"
            )
            M = np.pad(
                M, pad_width=pad_width, mode="constant", constant_values=pad_values
            )
            self.logger.debug(f"[max_pool] Padded matrix shape: {M.shape}")

        windows = sliding_window_view(M, window_shape=kernel_size)
        view = windows[::stride, ::stride]

        self.logger.debug(
            f"[max_pool] Pooling configuration: kernel={kernel_size}, stride={stride}, view_grid={view.shape[:2]}"
        )

        result = np.max(view, axis=(2, 3)).astype(np.uint8)

        if result.size == 0:
            self.logger.error(
                f"[max_pool] Data loss: Resulting matrix is empty for input_shape={M.shape}"
            )

        self.logger.info(f"[max_pool] Validated 1 items. Output shape={result.shape}")
        return result
