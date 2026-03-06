import logging
from dataclasses import dataclass, field
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


def convert_color_space(M: np.ndarray, to_gray: bool = True) -> np.ndarray:
    M_arr = np.asanyarray(M)

    if to_gray:
        if M_arr.ndim == 2:
            return M_arr.astype(np.uint8)
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return np.dot(M_arr[..., :3], weights).astype(np.uint8)

    if M_arr.ndim == 3:
        return M_arr.astype(np.uint8)
    return np.stack([M_arr] * 3, axis=-1).astype(np.uint8)


@dataclass
@class_autologger
class GrayScaleProcessing:
    logger: logging.Logger = field(init=False, repr=False)

    def convert_color_space(self, M: Mtx, to_gray: bool = True) -> Mtx:
        m_arr = np.asanyarray(M)
        self.logger.debug(
            f"[convert_color_space] Processing: input_shape={m_arr.shape}, to_gray={to_gray}"
        )

        result = convert_color_space(m_arr, to_gray=bool(to_gray))

        if to_gray and result.ndim != 2:
            self.logger.warning(f"Unexpected Gray dimension: {result.ndim}")
        elif not to_gray and result.shape[-1] != 3:
            self.logger.error(f"Failed RGB expansion: channels={result.shape[-1]}")

        self.logger.info(
            f"[convert_color_space] Validated items. Output shape={result.shape}"
        )
        return result
