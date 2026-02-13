from typing import Protocol, Tuple, TypeAlias

import numpy as np

from common_utils import class_autologger

Mtx: TypeAlias = np.ndarray
Size: TypeAlias = Tuple[int, int]


class ImageGeometryProtocol(Protocol):
    def resize(self, M: Mtx, new_size: Size = (28, 28)) -> Mtx: ...
    def prepare_standard_geometry(
        self, M: Mtx, target_size: Size = (28, 28), padding: int = 2, pad_value: int = 0
    ) -> Mtx: ...


@class_autologger
class ImageGeometry:
    def _upscale(self, M: Mtx, target_size: Size) -> Mtx:
        curr_h, curr_w = M.shape[:2]
        new_h, new_w = target_size

        scale_h = max(1, int(np.ceil(new_h / curr_h)))
        scale_w = max(1, int(np.ceil(new_w / curr_w)))

        self.logger.debug(
            f"[_upscale] Scaling factors: h={scale_h}, w={scale_w} to reach at least {target_size}"
        )
        return np.repeat(np.repeat(M, scale_h, axis=0), scale_w, axis=1)

    def _downscale_vectorized(self, M: Mtx, target_size: Size) -> Mtx:
        curr_h, curr_w = M.shape[:2]
        new_h, new_w = target_size

        h_f, w_f = curr_h // new_h, curr_w // new_w

        if h_f >= 1 and w_f >= 1:
            self.logger.debug(
                f"[_downscale_vectorized] Using vectorized mean pooling with factors: h_f={h_f}, w_f={w_f}"
            )
            return (
                M[: new_h * h_f, : new_w * w_f]
                .reshape(new_h, h_f, new_w, w_f)
                .mean(axis=(1, 3))
                .astype(np.uint8)
            )

        self.logger.debug(
            f"[_downscale_vectorized] Factors < 1. Falling back to zero-padding/cropping to {target_size}"
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
            self.logger.debug(
                f"[resize] Input size matches target size {new_size}. Skipping resize."
            )
            return M_arr

        if curr_h < new_h or curr_w < new_w:
            self.logger.debug(
                f"[resize] Input {M_arr.shape} is smaller than target {new_size}. Triggering upscale."
            )
            M_arr = self._upscale(M_arr, new_size)
            M_arr = self._downscale_vectorized(M_arr, new_size)
        else:
            self.logger.debug(
                f"[resize] Input {M_arr.shape} is larger than target {new_size}. Triggering downscale."
            )
            M_arr = self._downscale_vectorized(M_arr, new_size)

        return M_arr.astype(np.uint8)

    def prepare_standard_geometry(
        self, M: Mtx, target_size: Size = (28, 28), padding: int = 2, pad_value: int = 0
    ) -> Mtx:
        inner_h = target_size[0] - 2 * padding
        inner_w = target_size[1] - 2 * padding

        self.logger.debug(
            f"[prepare_standard_geometry] Target {target_size} with padding {padding} results in inner size ({inner_h}, {inner_w})"
        )
        resized = self.resize(M, (inner_h, inner_w))

        self.logger.info(
            f"[prepare_standard_geometry] Final matrix padded to {target_size} with value {pad_value}"
        )
        return np.pad(
            resized, pad_width=padding, mode="constant", constant_values=pad_value
        ).astype(np.uint8)
