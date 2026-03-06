import logging
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

import numpy as np
from common_utils import class_autologger
from PIL import Image

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]
FilePath: TypeAlias = str


class ImageConverterProtocol(Protocol):
    def get_channels_from_file(self, path: FilePath) -> MtxList: ...


def separate_channels(M: Mtx) -> MtxList:
    return [M[..., i] for i in range(3)]


def convert_image_to_matrix(path: FilePath) -> Mtx:
    with Image.open(path) as img:
        return np.array(img.convert("RGB")).astype(np.uint8)


@dataclass
@class_autologger
class ImageToMatrixConverter:
    logger: logging.Logger = field(init=False, repr=False)

    def _convert_image_to_matrix(self, path: FilePath) -> Mtx:
        matrix = convert_image_to_matrix(path)

        if matrix.shape[2] != 3:
            self.logger.warning(
                f"Unexpected channel count: {matrix.shape[2]} at {path}"
            )

        self.logger.debug(f"Converted {path} to RGB matrix: {matrix.shape}")
        return matrix

    def _separate_channels(self, M: Mtx) -> MtxList:
        self.logger.debug(f"Separating matrix with shape {M.shape}")
        return separate_channels(M)

    def get_channels_from_file(self, path: FilePath) -> MtxList:
        matrix = self._convert_image_to_matrix(path)

        if matrix is None or matrix.size == 0:
            self.logger.error(f"Data loss: Failed to create matrix from {path}")
            return []

        self.logger.info(f"Validated matrix creation from {path}")
        return self._separate_channels(matrix)
