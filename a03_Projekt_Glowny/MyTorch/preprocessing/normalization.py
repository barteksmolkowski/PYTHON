from dataclasses import dataclass
from typing import Tuple

import numpy as np

from MyTorch import ImageGray

from .base import NormalizationABC


def normalize(
    M: ImageGray, old_r: Tuple[float, float], new_r: Tuple[float, float]
) -> ImageGray:
    denominator: float = old_r[1] - old_r[0]
    if denominator == 0.0:
        return (np.zeros_like(M, dtype=np.float32) + new_r[0]).astype(np.float32)

    result = (M - old_r[0]) * (new_r[1] - new_r[0]) / denominator + new_r[0]
    return result.astype(np.float32)


def z_score_normalization(M: ImageGray, eps: float = 1e-8) -> ImageGray:
    mean_val: float = float(np.mean(M))
    std_val: float = float(np.std(M))
    result = (M - mean_val) / (std_val + eps)
    return result.astype(np.float32)


@dataclass(frozen=True, slots=True)
class Normalization(NormalizationABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    def normalize(
        self,
        M: ImageGray,
        old_r: Tuple[float, float] = (0.0, 255.0),
        new_r: Tuple[float, float] = (0.0, 1.0),
    ) -> ImageGray:
        return normalize(M, old_r, new_r)

    def z_score_normalization(self, M: ImageGray) -> ImageGray:
        return z_score_normalization(M)

    def process(
        self,
        M: ImageGray,
        use_z_score: bool = True,
        old_r: Tuple[float, float] = (0.0, 255.0),
        new_r: Tuple[float, float] = (0.0, 1.0),
    ) -> ImageGray:
        if use_z_score:
            return self.z_score_normalization(M)
        return self.normalize(M, old_r, new_r)
