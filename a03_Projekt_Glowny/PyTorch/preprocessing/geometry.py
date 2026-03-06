import logging
from dataclasses import dataclass, field
from typing import Protocol, Tuple, TypeAlias

import numpy as np
from common_utils import class_autologger

Mtx: TypeAlias = np.ndarray
Size: TypeAlias = Tuple[int, int]


class ImageGeometryProtocol(Protocol):
    def resize(self, M: Mtx, new_size: Size = (28, 28)) -> Mtx: ...

    def pad(self, M: Mtx, pad_value: int, padding: int) -> Mtx: ...

    def prepare_standard_geometry(
        self, M: Mtx, target_size: Size = (28, 28), padding: int = 2, pad_value: int = 0
    ) -> Mtx: ...


def pad(M: np.ndarray, pad_value: int, padding: int) -> np.ndarray:
    return np.pad(
        M, pad_width=padding, mode="constant", constant_values=pad_value
    ).astype(np.uint8)


def _upscale_bilinear(M: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    curr_h, curr_w = M.shape[:2]
    new_h, new_w = target_size

    if (curr_h, curr_w) == (new_h, new_w):
        return M

    y_coords = np.linspace(0, curr_h - 1, new_h)
    x_coords = np.linspace(0, curr_w - 1, new_w)

    y_low = np.floor(y_coords).astype(int)
    y_high = np.ceil(y_coords).astype(int)
    x_low = np.floor(x_coords).astype(int)
    x_high = np.ceil(x_coords).astype(int)

    y_high = np.clip(y_high, 0, curr_h - 1)
    x_high = np.clip(x_high, 0, curr_w - 1)

    y_weight = (y_coords - y_low)[:, np.newaxis]
    x_weight = (x_coords - x_low)[np.newaxis, :]

    top_left = M[y_low][:, x_low]
    top_right = M[y_low][:, x_high]
    bottom_left = M[y_high][:, x_low]
    bottom_right = M[y_high][:, x_high]

    top_row_mixed = top_left * (1 - x_weight) + top_right * x_weight
    bottom_row_mixed = bottom_left * (1 - x_weight) + bottom_right * x_weight

    result = top_row_mixed * (1 - y_weight) + bottom_row_mixed * y_weight

    return result.astype(np.uint8)


def _downscale_vectorized(M: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
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


def resize(M: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    curr_h, curr_w = M.shape[:2]
    new_h, new_w = new_size

    if (curr_h, curr_w) == (new_h, new_w):
        return M

    if curr_h < new_h or curr_w < new_w:
        M = _upscale_bilinear(M, (new_h, new_w))

    return _downscale_vectorized(M, (new_h, new_w))


def prepare_standard_geometry_logic(
    M: np.ndarray, target_size: Tuple[int, int], padding: int, pad_value: int
) -> np.ndarray:
    inner_h = target_size[0] - 2 * padding
    inner_w = target_size[1] - 2 * padding

    resized = resize(M, (inner_h, inner_w))
    return np.pad(
        resized, pad_width=padding, mode="constant", constant_values=pad_value
    ).astype(np.uint8)


@dataclass
@class_autologger
class ImageGeometry:
    logger: logging.Logger = field(init=False, repr=False)

    def pad(self, M: Mtx, pad_value: int, padding: int) -> Mtx:
        self.logger.debug(f"[pad] Shape {M.shape} -> padding {padding}")
        return pad(M, int(pad_value), int(padding))

    def resize(self, M: Mtx, new_size: Size = (28, 28)) -> Mtx:
        self.logger.debug(f"[resize] {M.shape} -> {new_size}")
        m_arr = np.asanyarray(M)
        result = resize(m_arr, (int(new_size[0]), int(new_size[1])))

        if result.shape[:2] != new_size:
            self.logger.warning(
                f"[resize] Shape mismatch: {result.shape} vs {new_size}"
            )
        return result

    def prepare_standard_geometry(
        self, M: Mtx, target_size: Size = (28, 28), padding: int = 2, pad_value: int = 0
    ) -> Mtx:
        self.logger.debug(
            f"[prepare_standard_geometry] Target {target_size}, pad {padding}"
        )

        m_arr = np.asanyarray(M)
        target_tuple = (int(target_size[0]), int(target_size[1]))

        result = prepare_standard_geometry_logic(
            M=m_arr,
            target_size=target_tuple,
            padding=int(padding),
            pad_value=int(pad_value),
        )
        return result
