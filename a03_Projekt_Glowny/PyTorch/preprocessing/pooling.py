import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple, TypeAlias

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


def max_pool_logic(
    matrix: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: int,
    pad_width: int,
    pad_values: int,
) -> np.ndarray:
    m_np = np.asanyarray(matrix, dtype=np.float32)
    if pad_width > 0:
        m_np = np.pad(
            m_np, pad_width=pad_width, mode="constant", constant_values=pad_values
        )

    windows = np.lib.stride_tricks.sliding_window_view(m_np, window_shape=kernel_size)
    view = windows[::stride, ::stride]
    return np.max(view, axis=(2, 3)).astype(np.uint8)


@dataclass
@class_autologger
class Pooling:
    logger: logging.Logger = field(init=False, repr=False)

    def max_pool(
        self,
        matrix: Mtx,
        kernel_size: Tuple[int, int] = (2, 2),
        stride: Optional[int] = None,
        pad_width: int = 0,
        pad_values: int = 0,
    ) -> Mtx:
        actual_stride = int(stride) if stride is not None else int(kernel_size[0])

        self.logger.debug(
            f"[max_pool] Config: kernel={kernel_size}, stride={actual_stride}, pad={pad_width}"
        )

        result = max_pool_logic(
            matrix=matrix,
            kernel_size=kernel_size,
            stride=actual_stride,
            pad_width=int(pad_width),
            pad_values=int(pad_values),
        )

        if result.size == 0:
            self.logger.error(f"[max_pool] Data loss: Empty output for {matrix.shape}")

        self.logger.info(f"[max_pool] Validated pooling. Output shape={result.shape}")
        return result
