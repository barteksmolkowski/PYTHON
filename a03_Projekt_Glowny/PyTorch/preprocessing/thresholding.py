import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, overload

import numpy as np
from common_utils import class_autologger
from numpy.lib.stride_tricks import sliding_window_view

from .decorators import parameter_complement

Mtx: TypeAlias = np.ndarray


class ThresholdingProtocol(Protocol):
    @overload
    def adaptive_threshold(
        self,
        matrix: Mtx,
        block_size: int = 5,
        c: int = 2,
        auto_params: Literal[True] = True,
    ) -> Mtx: ...

    @overload
    def adaptive_threshold(
        self, matrix: Mtx, block_size: int, c: int, auto_params: Literal[False]
    ) -> Mtx: ...

    def adaptive_threshold(
        self, matrix: Mtx, block_size: int = 5, c: int = 2, auto_params: bool = True
    ) -> Mtx: ...


def generate_gaussian_kernel(size: int) -> np.ndarray:
    v = size - np.abs(np.arange(-size + 1, size))
    return np.outer(v, v).astype(np.float32)


def adaptive_threshold_logic(
    matrix: np.ndarray, block_size: int, c: int, kernel: np.ndarray
) -> np.ndarray:
    pad_size = block_size // 2
    padded = np.pad(matrix, pad_width=pad_size, mode="reflect")
    windows = sliding_window_view(padded, (block_size, block_size))

    kernel_sum = np.sum(kernel)
    local_means = np.sum(windows * kernel, axis=(2, 3)) / kernel_sum

    return np.where(matrix > (local_means - c), 255, 0).astype(np.uint8)


@dataclass
@class_autologger
class Thresholding:
    logger: logging.Logger = field(init=False, repr=False)

    def _generate_gaussian_kernel(self, size: int) -> Mtx:
        self.logger.debug(f"[_generate_gaussian_kernel] Size={size}")
        return generate_gaussian_kernel(int(size))

    @parameter_complement
    def adaptive_threshold(
        self,
        matrix: Mtx,
        block_size: int = 5,
        c: int = 2,
        auto_params: bool = True,
    ) -> Mtx:
        m_arr = np.asanyarray(matrix)
        b_size = int(block_size)
        c_val = int(c)
        pad_size = b_size // 2

        if b_size % 2 == 0:
            self.logger.warning(f"[adaptive_threshold] Even block_size={b_size}")

        kernel = self._generate_gaussian_kernel(pad_size + 1)

        self.logger.debug(
            f"[adaptive_threshold] Config: block={b_size}, c={c_val}, auto={auto_params}"
        )

        result = adaptive_threshold_logic(
            matrix=m_arr, block_size=b_size, c=c_val, kernel=kernel
        )

        if result.shape != m_arr.shape:
            self.logger.error(
                f"[adaptive_threshold] Shape mismatch: {result.shape} vs {m_arr.shape}"
            )

        self.logger.info(f"[adaptive_threshold] Completed for shape={result.shape}")
        return result
