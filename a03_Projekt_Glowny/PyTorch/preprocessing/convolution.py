import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, List, Optional, Protocol, TypeAlias

import numpy as np
from common_utils import class_autologger
from numpy.lib.stride_tricks import sliding_window_view

from .geometry import ImageGeometry

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class GeometryProtocol(Protocol):
    def pad(self, M: Mtx, pad_value: int, padding: int) -> Mtx: ...


class ConvolutionProtocol(Protocol):
    def convolution_2d(self, M: Mtx, filters: Optional[MtxList] = None) -> MtxList: ...

    def apply_filters(self, channels: MtxList, filters: MtxList) -> MtxList: ...


def convolution_2d(M: np.ndarray, filters: List[np.ndarray]) -> List[np.ndarray]:
    if not filters:
        return [M]

    results = []
    M_np = np.asanyarray(M)

    for f in filters:
        f_np = np.asanyarray(f)
        fh, fw = f_np.shape
        windows = sliding_window_view(M_np, (fh, fw))
        conv_result = np.einsum("ij,klij->kl", f_np, windows)
        results.append(conv_result)

    return results


def apply_filters(
    channels: List[np.ndarray],
    filters: List[np.ndarray],
    pad_func: Callable[[np.ndarray, int, int], np.ndarray],
) -> List[np.ndarray]:
    final_results = []
    for f, channel in product(filters, channels):
        M_padded = pad_func(channel, -1, 1)
        conv_res = convolution_2d(M_padded, [f])
        final_results.extend(conv_res)
    return final_results


@dataclass
@class_autologger
class ConvolutionActions:
    geometry: Optional[GeometryProtocol] = None

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.geometry is None:
            self.logger.debug("geometry is None, selecting default ImageGeometry")
            self.geometry = ImageGeometry()

    def convolution_2d(self, M: Mtx, filters: Optional[MtxList] = None) -> MtxList:
        f_list = filters if filters is not None else []

        if not f_list:
            self.logger.debug("No filters provided, returning original matrix")
            return [M]

        self.logger.debug(
            f"Starting convolution: filters={len(f_list)}, shape={M.shape}"
        )

        for f in f_list:
            fh, fw = f.shape
            if fh > M.shape[0] or fw > M.shape[1]:
                self.logger.error(f"Kernel {f.shape} exceeds matrix {M.shape}")

        results = convolution_2d(M, f_list)
        self.logger.info(f"Validated {len(results)} convolution operations.")
        return results

    def apply_filters(self, channels: MtxList, filters: MtxList) -> MtxList:
        assert self.geometry is not None

        self.logger.info(
            f"Applying filters: {len(filters)}x{len(channels)}={len(filters) * len(channels)} ops"
        )

        results = apply_filters(
            channels=channels, filters=filters, pad_func=self.geometry.pad
        )

        self.logger.info(f"Validated {len(results)} filtered results.")
        return results
