import logging
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
    logger: logging.Logger

    def convert_color_space(self, M: Mtx, to_gray: bool = True) -> Mtx:
        M_arr = np.asanyarray(M)

        if to_gray:
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

            if M_arr.ndim == 2:
                self.logger.debug(
                    f"[convert_color_space] Input is already grayscale: ndim={M_arr.ndim}, shape={M_arr.shape}."
                )
                return M_arr.astype(np.uint8)

            self.logger.debug(
                f"[convert_color_space] Converting RGB to Gray: input_shape={M_arr.shape}, weights={weights}"
            )
            result = np.dot(M_arr[..., :3], weights).astype(np.uint8)

            if result.ndim != 2:
                self.logger.warning(
                    f"[convert_color_space] Unexpected output dimension after grayscale conversion: ndim={result.ndim}"
                )

            self.logger.info(
                f"[convert_color_space] Validated 1 items. Converted to shape={result.shape}"
            )
            return result

        else:
            if M_arr.ndim == 3:
                self.logger.debug(
                    f"[convert_color_space] Input is already 3D: ndim={M_arr.ndim}, shape={M_arr.shape}."
                )
                return M_arr.astype(np.uint8)

            self.logger.debug(
                f"[convert_color_space] Converting Gray to RGB via stacking: input_shape={M_arr.shape}"
            )
            result = np.stack([M_arr] * 3, axis=-1).astype(np.uint8)

            if result.shape[-1] != 3:
                self.logger.error(
                    f"[convert_color_space] Data loss: Failed to create 3-channel matrix. Current channels={result.shape[-1]}"
                )

            self.logger.info(
                f"[convert_color_space] Validated 1 items. Expanded to shape={result.shape}"
            )
            return result
