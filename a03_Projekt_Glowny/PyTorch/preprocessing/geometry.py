from typing import Protocol, Tuple, TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray
Size: TypeAlias = Tuple[int, int]


class ImageGeometryProtocol(Protocol):
    def resize(self, M: Mtx, new_size: Size = (28, 28)) -> Mtx: ...
    def prepare_standard_geometry(
        self, M: Mtx, target_size: Size = (28, 28), padding: int = 2, pad_value: int = 0
    ) -> Mtx: ...


class ImageGeometry:
    def _upscale(self, M: Mtx, target_size: Size) -> Mtx:
        curr_h, curr_w = M.shape[:2]
        new_h, new_w = target_size

        scale_h = max(1, int(np.ceil(new_h / curr_h)))
        scale_w = max(1, int(np.ceil(new_w / curr_w)))

        return np.repeat(np.repeat(M, scale_h, axis=0), scale_w, axis=1)

    def _downscale_vectorized(self, M: Mtx, target_size: Size) -> Mtx:
        curr_h, curr_w = M.shape[:2]
        new_h, new_w = target_size

        h_f, w_f = curr_h // new_h, curr_w // new_w

        if h_f >= 1 and w_f >= 1:
            return (
                M[: new_h * h_f, : new_w * w_f]
                .reshape(new_h, h_f, new_w, w_f)
                .mean(axis=(1, 3))
                .astype(np.uint8)
            )

        res = np.zeros((new_h, new_w), dtype=M.dtype)
        h_end, w_end = min(curr_h, new_h), min(curr_w, new_w)
        res[:h_end, :w_end] = M[:h_end, :w_end]
        return res.astype(np.uint8)

    def resize(self, M: Mtx, new_size: Size = (28, 28)) -> Mtx:
        M_arr = np.asanyarray(M)
        curr_h, curr_w = M_arr.shape[:2]
        new_h, new_w = new_size

        if (curr_h, curr_w) == new_size:
            return M_arr

        if curr_h < new_h or curr_w < new_w:
            M_arr = self._upscale(M_arr, new_size)
            M_arr = self._downscale_vectorized(M_arr, new_size)
        else:
            M_arr = self._downscale_vectorized(M_arr, new_size)

        return M_arr.astype(np.uint8)

    def prepare_standard_geometry(
        self, M: Mtx, target_size: Size = (28, 28), padding: int = 2, pad_value: int = 0
    ) -> Mtx:
        inner_h = target_size[0] - 2 * padding
        inner_w = target_size[1] - 2 * padding

        resized = self.resize(M, (inner_h, inner_w))

        return np.pad(
            resized, pad_width=padding, mode="constant", constant_values=pad_value
        ).astype(np.uint8)
