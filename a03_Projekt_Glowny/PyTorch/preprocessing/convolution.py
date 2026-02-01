from itertools import product
from typing import Optional, Protocol, TypeAlias

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .geometry import ImageGeometry

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class GeometryProtocol(Protocol):
    def pad(self, M: Mtx, pad_value: int, padding: int) -> Mtx: ...


class ConvolutionProtocol(Protocol):
    def convolution_2d(self, M: Mtx, filters: Optional[MtxList] = None) -> MtxList: ...

    def apply_filters(self, channels: MtxList, filters: MtxList) -> MtxList: ...


class ConvolutionActions:
    def __init__(self, geometry: Optional[GeometryProtocol] = None):
        self.geometry = geometry or ImageGeometry()

    def convolution_2d(self, M: Mtx, filters: Optional[MtxList] = None) -> MtxList:
        if not filters:
            return [M]

        M_np = np.asanyarray(M)
        results = []

        for f in filters:
            f_np = np.asanyarray(f)
            fh, fw = f_np.shape
            windows = sliding_window_view(M_np, (fh, fw))

            conv_result = np.einsum("ij,klij->kl", f_np, windows)
            results.append(conv_result)

        return results

    def apply_filters(self, channels: MtxList, filters: MtxList) -> MtxList:
        final_results = []

        for f, channel in product(filters, channels):
            M_padded = self.geometry.pad(channel, pad_value=-1, padding=1)

            conv_res = self.convolution_2d(M_padded, [f])
            final_results.extend(conv_res)

        return final_results
