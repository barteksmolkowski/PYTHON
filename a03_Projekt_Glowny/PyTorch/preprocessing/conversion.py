from typing import Protocol, TypeAlias

import numpy as np
from PIL import Image

from common_utils import class_autologger

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]
FilePath: TypeAlias = str


class ImageConverterProtocol(Protocol):
    def get_channels_from_file(self, path: FilePath) -> MtxList: ...

@class_autologger
class ImageToMatrixConverter:
    def _convert_image_to_matrix(self, path: FilePath) -> Mtx:
        with Image.open(path) as img:
            matrix = np.array(img.convert("RGB")).astype(np.uint8)
            self.logger.debug(f"[_convert_image_to_matrix] Converted image from {path} to RGB matrix with shape {matrix.shape}")
            return matrix

    def _separate_channels(self, M: Mtx) -> MtxList:
        channels = [M[..., i] for i in range(3)]
        self.logger.debug(f"[_separate_channels] Separated matrix into {len(channels)} channels")
        return channels

    def get_channels_from_file(self, path: FilePath) -> MtxList:
        matrix = self._convert_image_to_matrix(path)
        self.logger.info(f"[get_channels_from_file] Matrix successfully created from {path}")
        return self._separate_channels(matrix)
