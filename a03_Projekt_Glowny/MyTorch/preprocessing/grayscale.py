from dataclasses import dataclass
from typing import Literal, Union, overload

import numpy as np

from MyTorch import ImageGray, ImageRGB

from .base import GrayScaleABC


@overload
def convert_color_space(M: ImageRGB, to_gray: Literal[True]) -> ImageGray: ...


@overload
def convert_color_space(M: ImageGray, to_gray: Literal[False]) -> ImageRGB: ...


def convert_color_space(
    M: Union[ImageGray, ImageRGB], to_gray: bool = True
) -> Union[ImageGray, ImageRGB]:
    if to_gray:
        if M.ndim == 2:
            return M.astype(np.float32)
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return np.dot(M.transpose(1, 2, 0), weights).astype(np.float32)

    if M.ndim == 3:
        return M.astype(np.float32)

    return np.stack([M] * 3, axis=0).astype(np.float32)


@dataclass(frozen=True, slots=True)
class GrayScaleProcessing(GrayScaleABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    def convert_color_space(
        self, M: Union[ImageGray, ImageRGB], to_gray: bool = True
    ) -> Union[ImageGray, ImageRGB]:
        return convert_color_space(M, to_gray=to_gray)
