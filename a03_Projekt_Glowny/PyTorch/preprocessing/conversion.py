from typing import Protocol, TypeAlias

import numpy as np
from PIL import Image

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]
FilePath: TypeAlias = str


class ImageConverterProtocol(Protocol):
    def get_channels_from_file(self, path: FilePath) -> MtxList: ...


class ImageToMatrixConverter:
    def _convert_image_to_matrix(self, path: FilePath) -> Mtx:
        with Image.open(path) as img:
            return np.array(img.convert("RGB")).astype(np.uint8)

    def _separate_channels(self, M: Mtx) -> MtxList:
        return [M[..., i] for i in range(3)]

    def get_channels_from_file(self, path: FilePath) -> MtxList:
        matrix = self._convert_image_to_matrix(path)
        return self._separate_channels(matrix)
