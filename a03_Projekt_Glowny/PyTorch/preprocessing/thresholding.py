from typing import Literal, Protocol, overload

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .decorators import parameter_complement

Mtx = np.ndarray


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


class Thresholding:
    def _generate_gaussian_kernel(self, size: int) -> Mtx:
        v = size - np.abs(np.arange(-size + 1, size))
        return np.outer(v, v).astype(np.float32)

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
        
        kernel = self._generate_gaussian_kernel(pad_size + 1)
        
        padded = np.pad(M, pad_width=pad_size, mode="reflect")

        windows = sliding_window_view(padded, (block_size, block_size))

        local_means = np.sum(windows * kernel, axis=(2, 3)) / np.sum(kernel)

        return np.where(M > (local_means - c), 255, 0).astype(np.uint8)
