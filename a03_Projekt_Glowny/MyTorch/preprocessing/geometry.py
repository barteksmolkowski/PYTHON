from dataclasses import dataclass

import numpy as np

from MyTorch import ImageGray, Padded, Shape

from .base import ImageGeometryABC


def pad(M: ImageGray, pad_value: int, padding: int) -> Padded:
    return np.pad(
        M, pad_width=padding, mode="constant", constant_values=float(pad_value)
    ).astype(np.float32)


def _upscale_bilinear(M: ImageGray, target_size: Shape) -> ImageGray:
    curr_h, curr_w = M.shape
    new_h, new_w = target_size

    if (curr_h, curr_w) == (new_h, new_w):
        return M

    y_coords = np.linspace(0, curr_h - 1, new_h)
    x_coords = np.linspace(0, curr_w - 1, new_w)

    y_low = np.floor(y_coords).astype(int)
    y_high = np.clip(np.ceil(y_coords).astype(int), 0, curr_h - 1)
    x_low = np.floor(x_coords).astype(int)
    x_high = np.clip(np.ceil(x_coords).astype(int), 0, curr_w - 1)

    y_weight = (y_coords - y_low)[:, np.newaxis]
    x_weight = (x_coords - x_low)[np.newaxis, :]

    top_left = M[y_low][:, x_low]
    top_right = M[y_low][:, x_high]
    bottom_left = M[y_high][:, x_low]
    bottom_right = M[y_high][:, x_high]

    top_row_mixed = top_left * (1.0 - x_weight) + top_right * x_weight
    bottom_row_mixed = bottom_left * (1.0 - x_weight) + bottom_right * x_weight

    result = top_row_mixed * (1.0 - y_weight) + bottom_row_mixed * y_weight

    return result.astype(np.float32)


def _downscale_vectorized(M: ImageGray, target_size: Shape) -> ImageGray:
    curr_h, curr_w = M.shape
    new_h, new_w = target_size
    h_f, w_f = curr_h // new_h, curr_w // new_w

    if h_f >= 1 and w_f >= 1:
        return (
            M[: new_h * h_f, : new_w * w_f]
            .reshape(new_h, h_f, new_w, w_f)
            .mean(axis=(1, 3))
            .astype(np.float32)
        )

    res = np.zeros((new_h, new_w), dtype=np.float32)
    h_end, w_end = min(curr_h, new_h), min(curr_w, new_w)
    res[:h_end, :w_end] = M[:h_end, :w_end]
    return res.astype(np.float32)


def resize(M: ImageGray, new_size: Shape) -> ImageGray:
    curr_h, curr_w = M.shape
    new_h, new_w = new_size

    if (curr_h, curr_w) == (new_h, new_w):
        return M

    if curr_h < new_h or curr_w < new_w:
        M = _upscale_bilinear(M, (new_h, new_w))

    return _downscale_vectorized(M, (new_h, new_w))


def prepare_standard_geometry_logic(
    M: ImageGray, target_size: Shape, padding: int, pad_value: int
) -> Padded:
    inner_h: int = int(target_size[0] - 2 * padding)
    inner_w: int = int(target_size[1] - 2 * padding)

    resized = resize(M, (inner_h, inner_w))
    return np.pad(
        resized, pad_width=padding, mode="constant", constant_values=float(pad_value)
    ).astype(np.float32)


@dataclass(frozen=True, slots=True)
class ImageGeometry(ImageGeometryABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    def pad(self, M: ImageGray, pad_value: int, padding: int) -> Padded:
        return pad(M, pad_value, padding)

    def resize(self, M: ImageGray, new_size: Shape) -> ImageGray:
        return resize(M, new_size)

    def prepare_standard_geometry(
        self, M: ImageGray, target_size: Shape, padding: int = 2, pad_value: int = 0
    ) -> Padded:
        return prepare_standard_geometry_logic(
            M=M,
            target_size=target_size,
            padding=padding,
            pad_value=pad_value,
        )
