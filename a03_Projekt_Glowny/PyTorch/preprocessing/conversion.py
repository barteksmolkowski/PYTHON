import logging
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
    logger: logging.Logger

    def _convert_image_to_matrix(self, path: FilePath) -> Mtx:
        with Image.open(path) as img:
            matrix = np.array(img.convert("RGB")).astype(np.uint8)

            if matrix.shape[2] != 3:
                self.logger.warning(
                    f"[_convert_image_to_matrix] Unexpected channel count: channels={matrix.shape[2]} for path='{path}'"
                )

            self.logger.debug(
                f"[_convert_image_to_matrix] Converted image to RGB matrix: shape={matrix.shape}, path='{path}'"
            )
            return matrix

    def _separate_channels(self, M: Mtx) -> MtxList:
        channels = [M[..., i] for i in range(3)]
        self.logger.debug(
            f"[_separate_channels] Separated matrix (shape={M.shape}) into {len(channels)} channels"
        )
        return channels

    def get_channels_from_file(self, path: FilePath) -> MtxList:
        matrix = self._convert_image_to_matrix(path)

        if matrix is None or matrix.size == 0:
            self.logger.error(
                f"[get_channels_from_file] Data loss: Failed to create matrix from path='{path}'"
            )

        self.logger.info(
            f"[get_channels_from_file] Validated 1 items. Matrix created from path='{path}'"
        )
        return self._separate_channels(matrix)
