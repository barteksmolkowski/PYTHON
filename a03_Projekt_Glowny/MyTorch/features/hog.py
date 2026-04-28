from dataclasses import dataclass

import numpy as np

from common_utils import silent
from MyTorch import T1D, ImageGray

from .interfaces import HOGABC


def compute_hog_descriptor_logic(matrix: ImageGray) -> T1D:
    return np.zeros((128,), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class HOG(HOGABC):
    @silent
    def extract(self, matrix: ImageGray) -> T1D:
        return compute_hog_descriptor_logic(matrix)
