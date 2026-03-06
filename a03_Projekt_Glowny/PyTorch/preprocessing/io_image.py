import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol, TypeAlias, Union, overload

import numpy as np
from common_utils import class_autologger
from PIL import Image

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


def open_image(path: str) -> tuple[np.ndarray, int, int]:
    from PIL import Image

    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        width, height = img_rgb.size
        return np.array(img_rgb).astype(np.uint8), width, height


def save_image(data: np.ndarray, path: str) -> None:
    from PIL import Image

    array = np.asanyarray(data).astype(np.uint8)
    mode = "RGB" if array.ndim == 3 else "L"
    img_pil = Image.fromarray(array, mode=mode)
    img_pil.save(path)


@dataclass
@class_autologger
class ImageHandler:
    logger: logging.Logger = field(init=False, repr=False)

    def open_image(self, path: FilePath) -> tuple[Mtx, int, int]:
        str_path = str(path)
        array, width, height = open_image(str_path)
        self.logger.debug(f"Loaded: {str_path}, {width}x{height}")
        self.logger.info(f"Validated 1 items. Image loaded from '{str_path}'")
        return array, int(width), int(height)

    def save(self, data: ImgData, path: FilePath) -> None:
        str_path = str(path)
        array = np.asanyarray(data).astype(np.uint8)

        if array.size == 0:
            self.logger.error(f"Data loss: Empty matrix for '{str_path}'")
            return

        self.logger.debug(f"Saving to '{str_path}': shape={array.shape}")
        save_image(array, str_path)
        self.logger.info(f"Validated 1 items. Image saved to '{str_path}'")

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
        str_path = str(path)
        if is_save_mode:
            if data is None:
                self.logger.error(f"Validation error: data is None for '{str_path}'")
                raise ValueError("The 'data' parameter is required in write mode!")

            self.logger.debug(f"Switching to SAVE mode: '{str_path}'")
            self.save(data, path)
            return None

        self.logger.debug(f"Switching to READ mode: '{str_path}'")
        matrix, _, _ = self.open_image(path)
        return matrix
