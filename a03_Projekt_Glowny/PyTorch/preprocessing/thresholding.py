import logging
from typing import Literal, Protocol, TypeAlias, overload

import numpy as np
from common_utils import class_autologger
from numpy.lib.stride_tricks import sliding_window_view

from .decorators import parameter_complement

Mtx: TypeAlias = np.ndarray


class ThresholdingProtocol(Protocol):
    @overload
    def adaptive_threshold(
        self, M: Mtx, block_size: int = 0, c: int = 0, auto_params: Literal[True] = True
    ) -> Mtx: ...

    @overload
    def adaptive_threshold(
        self, M: Mtx, block_size: int, c: int, auto_params: Literal[False]
    ) -> Mtx: ...

    def adaptive_threshold(
        self, M: Mtx, block_size: int = 5, c: int = 2, auto_params: bool = True
    ) -> Mtx: ...


@class_autologger
class Thresholding:
    logger: logging.Logger

    def _generate_gaussian_kernel(self, size: int) -> Mtx:
        v = size - np.abs(np.arange(-size + 1, size))
        kernel = np.outer(v, v).astype(np.float32)

        self.logger.debug(
            f"[_generate_gaussian_kernel] Generated kernel: size={size}, shape={kernel.shape}"
        )
        return kernel

    @parameter_complement
    def adaptive_threshold(
        self,
        matrix: Mtx,
        block_size: int = 5,
        c: int = 2,
        auto_params: bool = True,
    ) -> Mtx:
        M = np.asanyarray(matrix)
        pad_size = block_size // 2

        if block_size % 2 == 0:
            self.logger.warning(
                f"[adaptive_threshold] Even block_size={block_size} detected. Logic might be shifted."
            )

        kernel = self._generate_gaussian_kernel(pad_size + 1)
        self.logger.debug(
            f"[adaptive_threshold] Configuration: block_size={block_size}, c={c}, pad_size={pad_size}, auto_params={auto_params}"
        )

        padded = np.pad(M, pad_width=pad_size, mode="reflect")
        self.logger.debug(
            f"[adaptive_threshold] Matrix padded: mode='reflect', padded_shape={padded.shape}"
        )

        windows = sliding_window_view(padded, (block_size, block_size))

        kernel_sum = np.sum(kernel)
        local_means = np.sum(windows * kernel, axis=(2, 3)) / kernel_sum

        self.logger.info(
            f"[adaptive_threshold] Validated local means calculation with kernel_sum={kernel_sum:.2f}"
        )

        result = np.where(M > (local_means - c), 255, 0).astype(np.uint8)

        if result.shape != M.shape:
            self.logger.error(
                f"[adaptive_threshold] Shape mismatch: result={result.shape} vs original={M.shape}"
            )

        self.logger.info(
            f"[adaptive_threshold] Validated 1 items. Thresholding complete for shape={result.shape}."
        )
        return result
