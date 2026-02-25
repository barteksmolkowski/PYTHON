import logging
from typing import Literal, Optional, Protocol, TypeAlias, Union, overload

import numpy as np
from PIL import Image

from common_utils import class_autologger

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]
FilePath: TypeAlias = str
ImgData: TypeAlias = Union[Mtx, MtxList]


class ImageHandlerProtocol(Protocol):
    def open_image(self, path: FilePath) -> tuple[Mtx, int, int]: ...

    @overload
    def save(self, data: Mtx, path: FilePath) -> None: ...
    @overload
    def save(self, data: MtxList, path: FilePath) -> None: ...
    def save(self, data: ImgData, path: FilePath) -> None: ...

    @overload
    def handle_file(
        self, path: FilePath, data: None = None, is_save_mode: Literal[False] = False
    ) -> Mtx: ...

    @overload
    def handle_file(
        self, path: FilePath, data: Mtx, is_save_mode: Literal[True]
    ) -> None: ...

    def handle_file(
        self, path: FilePath, data: Optional[Mtx] = None, is_save_mode: bool = False
    ) -> Optional[Mtx]: ...


@class_autologger
class ImageHandler:
    logger: logging.Logger

    def open_image(self, path: FilePath) -> tuple[Mtx, int, int]:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            width, height = img_rgb.size
            array = np.array(img_rgb).astype(np.uint8)

            self.logger.debug(
                f"[open_image] Loaded resource: path='{path}', size={width}x{height}, mode='{img.mode}'"
            )
            self.logger.info(
                f"[open_image] Validated 1 items. Image loaded from '{path}'"
            )
            return array, width, height

    def save(self, data: ImgData, path: FilePath) -> None:
        array = np.asanyarray(data).astype(np.uint8)
        mode = "RGB" if array.ndim == 3 else "L"

        if array.size == 0:
            self.logger.error(
                f"[save] Data loss: Attempting to save empty matrix to '{path}'"
            )

        self.logger.debug(
            f"[save] Saving to path='{path}': shape={array.shape}, pil_mode='{mode}'"
        )

        img_pil = Image.fromarray(array, mode=mode)
        img_pil.save(path)
        self.logger.info(f"[save] Validated 1 items. Image saved to '{path}'")

    @overload
    def handle_file(
        self, path: FilePath, data: None = None, is_save_mode: Literal[False] = False
    ) -> Mtx: ...

    @overload
    def handle_file(
        self, path: FilePath, data: Mtx, is_save_mode: Literal[True]
    ) -> None: ...

    def handle_file(
        self, path: FilePath, data: Optional[Mtx] = None, is_save_mode: bool = False
    ) -> Optional[Mtx]:
        if is_save_mode:
            if data is None:
                self.logger.error(
                    f"[handle_file] Validation error: data is None while is_save_mode={is_save_mode} for path='{path}'"
                )
                raise ValueError(
                    "[ERROR] The 'data' parameter is required in write mode!"
                )

            self.logger.debug(f"[handle_file] Switching to SAVE mode: path='{path}'")
            self.save(data, path)
            return None

        self.logger.debug(f"[handle_file] Switching to READ mode: path='{path}'")
        return self.open_image(path)[0]
