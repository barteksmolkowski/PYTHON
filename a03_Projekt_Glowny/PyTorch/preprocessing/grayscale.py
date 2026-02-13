from typing import Literal, Protocol, TypeAlias, overload

import numpy as np

from common_utils import class_autologger

Mtx: TypeAlias = np.ndarray


class GrayScaleProtocol(Protocol):
    @overload
    def convert_color_space(self, M: Mtx, to_gray: Literal[True] = True) -> Mtx: ...

    @overload
    def convert_color_space(self, M: Mtx, to_gray: Literal[False]) -> Mtx: ...

    def convert_color_space(self, M: Mtx, to_gray: bool = True) -> Mtx: ...


@class_autologger
class GrayScaleProcessing:
    def convert_color_space(self, M: Mtx, to_gray: bool = True) -> Mtx:
        M_arr = np.asanyarray(M)

        if to_gray:
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

            if M_arr.ndim == 2:
                self.logger.debug(
                    f"[convert_color_space] Image is already grayscale (ndim={M_arr.ndim}). Skipping dot product."
                )
                return M_arr.astype(np.uint8)

            self.logger.debug(
                f"[convert_color_space] Converting RGB to Gray using weights {weights}. Input shape: {M_arr.shape}"
            )
            return np.dot(M_arr[..., :3], weights).astype(np.uint8)

        else:
            if M_arr.ndim == 3:
                self.logger.debug(
                    f"[convert_color_space] Image is already in RGB-like format (ndim={M_arr.ndim}). Skipping stack."
                )
                return M_arr.astype(np.uint8)

            self.logger.debug(
                f"[convert_color_space] Converting Gray to RGB by stacking channels. Input shape: {M_arr.shape}"
            )
            return np.stack([M_arr] * 3, axis=-1).astype(np.uint8)
