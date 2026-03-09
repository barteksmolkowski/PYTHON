import logging
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

import numpy as np
from common_utils import class_autologger

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class EdgeDetectorProtocol(Protocol):
    def apply(self, matrix: Mtx) -> Mtx: ...


def apply_sobel_filter_logic(matrix: Mtx) -> Mtx:
    return matrix


def apply_prewitt_filter_logic(matrix: Mtx) -> Mtx:
    return matrix


@dataclass
@class_autologger
class Sobel:
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply(self, matrix: Mtx) -> Mtx:
        return apply_sobel_filter_logic(matrix)


@class_autologger
@dataclass
class Prewitt:
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply(self, matrix: Mtx) -> Mtx:
        return apply_prewitt_filter_logic(matrix)
