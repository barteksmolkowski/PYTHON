from dataclasses import dataclass, field
from itertools import product
from typing import Callable, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from MyTorch import T2D, ImageGray, Padded

from .base import ConvolutionABC
from .geometry import ImageGeometry
from .protocols import ImageGeometryProtocol


def convolution_2d(M: Padded, filters: List[T2D]) -> List[ImageGray]:
    if not filters:
        return [M.astype(np.float32)]

    results: List[ImageGray] = list()

    for f in filters:
        fh, fw = f.shape
        windows = sliding_window_view(M, (fh, fw))
        conv_result = np.einsum("ij,hwij->hw", f, windows).astype(np.float32)
        results.append(conv_result)

    return results


def apply_filters(
    channels: List[ImageGray],
    filters: List[T2D],
    pad_func: Callable[[ImageGray, int, int], Padded],
) -> List[ImageGray]:
    final_results: List[ImageGray] = list()
    for f, channel in product(filters, channels):
        M_padded = pad_func(channel, 0, 0)
        conv_res = convolution_2d(M_padded, [f])
        final_results.extend(conv_res)
    return final_results


@dataclass(frozen=True, slots=True)
class ConvolutionActions(ConvolutionABC):
    geometry: ImageGeometryProtocol = field(default_factory=ImageGeometry)

    def __post_init__(self) -> None:
        super().__post_init__()

    def convolution_2d(
        self, M: ImageGray, filters: List[T2D] = list()
    ) -> List[ImageGray]:
        return convolution_2d(M, filters)

    def apply_filters(
        self, channels: List[ImageGray], filters: List[T2D] = list()
    ) -> List[ImageGray]:
        return apply_filters(
            channels=channels, filters=filters, pad_func=self.geometry.pad
        )
