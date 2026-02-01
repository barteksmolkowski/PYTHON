from typing import Literal, Protocol, TypeAlias, overload

import numpy as np

Mtx: TypeAlias = np.ndarray


class GrayScaleProtocol(Protocol):
    @overload
    def convert_color_space(self, M: Mtx, to_gray: Literal[True] = True) -> Mtx: ...

    @overload
    def convert_color_space(self, M: Mtx, to_gray: Literal[False]) -> Mtx: ...

    def convert_color_space(self, M: Mtx, to_gray: bool = True) -> Mtx: ...


class GrayScaleProcessing:
    def convert_color_space(self, M: Mtx, to_gray: bool = True) -> Mtx:
        M_arr = np.asanyarray(M)

        if to_gray:
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

            if M_arr.ndim == 2:
                return M_arr.astype(np.uint8)

            return np.dot(M_arr[..., :3], weights).astype(np.uint8)

        else:
            if M_arr.ndim == 3:
                return M_arr.astype(np.uint8)

            return np.stack([M_arr] * 3, axis=-1).astype(np.uint8)
