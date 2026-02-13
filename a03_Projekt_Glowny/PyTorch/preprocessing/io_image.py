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
    def open_image(self, path: FilePath) -> tuple[Mtx, int, int]:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            width, height = img_rgb.size
            array = np.array(img_rgb).astype(np.uint8)
            self.logger.debug(
                f"[open_image] Loaded image from {path} (Size: {width}x{height}, Mode: {img.mode})"
            )
            return array, width, height

    def save(self, data: ImgData, path: FilePath) -> None:
        array = np.asanyarray(data).astype(np.uint8)
        mode = "RGB" if array.ndim == 3 else "L"

        self.logger.debug(
            f"[save] Preparing to save image to {path} (Shape: {array.shape}, Mode: {mode})"
        )

        img_pil = Image.fromarray(array, mode=mode)
        img_pil.save(path)

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
                    f"[handle_file] Failed to save: data is None for path {path}"
                )
                raise ValueError(
                    "[ERROR] The 'data' parameter is required in write mode!"
                )

            self.logger.info(f"[handle_file] Entering SAVE mode for path: {path}")
            self.save(data, path)
            return None

        self.logger.info(f"[handle_file] Entering READ mode for path: {path}")
        return self.open_image(path)[0]
