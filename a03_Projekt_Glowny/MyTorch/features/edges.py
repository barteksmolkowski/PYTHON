from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from MyTorch import ImageGray

from .interfaces import EdgeDetectorABC

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


def apply_prewitt_filter_logic(matrix: ImageGray) -> ImageGray:
    return matrix


def apply_sobel_filter_logic(matrix: ImageGray) -> ImageGray:
    return matrix


@dataclass(frozen=True, slots=True)
class Sobel(EdgeDetectorABC):
    def apply(self, matrix: ImageGray) -> ImageGray:
        return apply_sobel_filter_logic(matrix)


@dataclass(frozen=True, slots=True)
class Prewitt(EdgeDetectorABC):
    def apply(self, matrix: ImageGray) -> ImageGray:
        return apply_prewitt_filter_logic(matrix)
