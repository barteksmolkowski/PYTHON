from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

from MyTorch import FilePath, ImageGray, ImageRGB

from .base import ConverterABC


def separate_channels(M: ImageRGB) -> List[ImageGray]:
    return [M[i, :, :].astype(np.float32) for i in range(3)]


def convert_image_to_matrix(path: FilePath) -> ImageRGB:
    with Image.open(path) as img:
        return np.array(img.convert("RGB")).transpose(2, 0, 1).astype(np.float32)


@dataclass(frozen=True, slots=True)
class ImageToMatrixConverter(ConverterABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    def get_channels_from_file(self, path: FilePath) -> List[ImageGray]:
        matrix: ImageRGB = self._convert_image_to_matrix(path)
        return self._separate_channels(matrix)

    def _convert_image_to_matrix(self, path: FilePath) -> ImageRGB:
        return convert_image_to_matrix(path)

    def _separate_channels(self, M: ImageRGB) -> List[ImageGray]:
        return separate_channels(M)
