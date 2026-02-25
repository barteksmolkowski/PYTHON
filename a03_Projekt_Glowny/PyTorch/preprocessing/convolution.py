from itertools import product
from typing import Optional, Protocol, TypeAlias

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import logging
from common_utils import class_autologger

from .geometry import ImageGeometry

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class GeometryProtocol(Protocol):
    def pad(self, M: Mtx, pad_value: int, padding: int) -> Mtx: ...


class ConvolutionProtocol(Protocol):
    def convolution_2d(self, M: Mtx, filters: Optional[MtxList] = None) -> MtxList: ...

    def apply_filters(self, channels: MtxList, filters: MtxList) -> MtxList: ...


@class_autologger
class ConvolutionActions:
    logger: logging.Logger

    def __init__(self, geometry: Optional[GeometryProtocol] = None):
        if geometry is None:
            self.logger.debug(
                "[__init__] geometry is None, selecting default ImageGeometry"
            )
        self.geometry = geometry or ImageGeometry()

    def convolution_2d(self, M: Mtx, filters: Optional[MtxList] = None) -> MtxList:
        if not filters:
            self.logger.debug(
                "[convolution_2d] No filters provided (filters=None/empty), returning original matrix in list"
            )
            return [M]

        M_np = np.asanyarray(M)
        results = []

        self.logger.debug(
            f"[convolution_2d] Starting convolution: filters_count={len(filters)}, matrix_shape={M_np.shape}"
        )

        for i, f in enumerate(filters):
            f_np = np.asanyarray(f)
            fh, fw = f_np.shape

            if fh > M_np.shape[0] or fw > M_np.shape[1]:
                self.logger.error(
                    f"[convolution_2d] Kernel size ({fh}, {fw}) exceeds matrix size {M_np.shape}. Data loss potential."
                )

            windows = sliding_window_view(M_np, (fh, fw))

            self.logger.debug(
                f"[convolution_2d] Processing filter index={i} with kernel_size=({fh}, {fw})"
            )

            conv_result = np.einsum("ij,klij->kl", f_np, windows)
            results.append(conv_result)

        self.logger.info(
            f"[convolution_2d] Validated {len(results)} convolution operations."
        )
        return results

    def apply_filters(self, channels: MtxList, filters: MtxList) -> MtxList:
        final_results = []

        num_filters = len(filters)
        num_channels = len(channels)
        total_operations = num_filters * num_channels

        self.logger.info(
            f"[apply_filters] Preparing to apply filters: filters_count={num_filters}, channels_count={num_channels} (Total operations={total_operations})"
        )

        for f, channel in product(filters, channels):
            M_padded = self.geometry.pad(channel, pad_value=-1, padding=1)

            self.logger.debug(
                f"[apply_filters] Channel padded: original_shape={channel.shape}, padded_shape={M_padded.shape}"
            )

            conv_res = self.convolution_2d(M_padded, [f])
            final_results.extend(conv_res)

        self.logger.info(
            f"[apply_filters] Validated {len(final_results)} filtered results."
        )
        return final_results
