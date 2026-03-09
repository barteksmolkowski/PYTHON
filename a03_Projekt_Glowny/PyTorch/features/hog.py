import logging
from dataclasses import dataclass, field
from typing import List, Protocol, TypeAlias

import numpy as np
from common_utils import class_autologger, silent

Mtx: TypeAlias = np.ndarray


class HOGProtocol(Protocol):
    def extract(self, matrix: Mtx) -> list[float]: ...


def compute_hog_descriptor_logic(matrix: Mtx) -> List[float]:
    return [0.0]


@class_autologger
@dataclass
class HOG:
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @silent
    def extract(self, matrix: Mtx) -> List[float]:
        return compute_hog_descriptor_logic(matrix)
