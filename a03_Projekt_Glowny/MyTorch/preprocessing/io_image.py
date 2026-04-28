from dataclasses import dataclass
from typing import Tuple, Union, cast

import numpy as np
from PIL import Image

from MyTorch import FilePath, ImageGray, ImageRGB

from .base import ImageHandlerABC


def open_image(path: FilePath) -> Tuple[ImageRGB, int, int]:
    from PIL import Image

    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        width, height = img_rgb.size
        matrix = np.array(img_rgb).transpose(2, 0, 1).astype(np.float32)
        return cast(ImageRGB, matrix), width, height


def save_image(data: Union[ImageGray, ImageRGB], path: FilePath) -> None:
    if data.ndim == 3:
        array = data.transpose(1, 2, 0).astype(np.uint8)
        mode = "RGB"
    else:
        array = data.astype(np.uint8)
        mode = "L"

    img_pil = Image.fromarray(array, mode=mode)
    img_pil.save(path)


@dataclass(frozen=True, slots=True)
class ImageHandler(ImageHandlerABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    def open_image(self, path: FilePath) -> Tuple[ImageRGB, int, int]:
        return open_image(path)

    def save(self, data: Union[ImageGray, ImageRGB], path: FilePath) -> None:
        save_image(data, path)

    def handle_file(
        self,
        path: FilePath,
        data: Union[ImageGray, ImageRGB],
        is_save_mode: bool = False,
    ) -> Union[ImageGray, ImageRGB]:
        if is_save_mode:
            self.save(data, path)
            return data

        matrix, _, _ = self.open_image(path)
        return matrix
