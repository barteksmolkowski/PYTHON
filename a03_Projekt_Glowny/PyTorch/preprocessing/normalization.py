import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Tuple, TypeAlias, cast, overload

import numpy as np
from common_utils import class_autologger

Mtx: TypeAlias = np.ndarray
Range: TypeAlias = tuple[float, float]


class NormalizationProtocol(Protocol):
    def normalize(
        self, M: Mtx, old_r: Range = (0, 255), new_r: Range = (0, 1)
    ) -> Mtx: ...

    def z_score_normalization(self, M: Mtx) -> Mtx: ...

    @overload
    def process(self, M: Mtx, use_z_score: Literal[True]) -> Mtx: ...

    @overload
    def process(
        self, M: Mtx, use_z_score: Literal[False], old_r: Range, new_r: Range
    ) -> Mtx: ...

    def process(
        self, M: Mtx, use_z_score: bool, old_r: Any = None, new_r: Any = None
    ) -> Mtx: ...


def normalize(
    M: np.ndarray, old_r: Tuple[float, float], new_r: Tuple[float, float]
) -> np.ndarray:
    denominator = old_r[1] - old_r[0]
    if denominator == 0:
        return np.zeros_like(M, dtype=np.float32) + new_r[0]
    return (M - old_r[0]) * (new_r[1] - new_r[0]) / denominator + new_r[0]


def z_score_normalization(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean_val, std_val = np.mean(M), np.std(M)
    return (M - mean_val) / (std_val + eps)


@dataclass
@class_autologger
class Normalization:
    logger: logging.Logger = field(init=False, repr=False)

    def normalize(self, M: Mtx, old_r: Range = (0, 255), new_r: Range = (0, 1)) -> Mtx:
        self.logger.debug(f"[normalize] Scaling {old_r} -> {new_r}")

        o_r = cast(Tuple[float, float], old_r)
        n_r = cast(Tuple[float, float], new_r)

        return normalize(np.asanyarray(M, dtype=np.float32), o_r, n_r)

    def z_score_normalization(self, M: Mtx) -> Mtx:
        m_np = np.asanyarray(M, dtype=np.float32)
        std_val = float(np.std(m_np))
        if std_val < 1e-8:
            self.logger.warning(
                f"[z_score_normalization] Low variance: std={std_val:.8f}"
            )
        return z_score_normalization(m_np)

    @overload
    def process(self, M: Mtx, use_z_score: Literal[True]) -> Mtx: ...

    @overload
    def process(
        self, M: Mtx, use_z_score: Literal[False], old_r: Range, new_r: Range
    ) -> Mtx: ...

    def process(
        self,
        M: Mtx,
        use_z_score: bool = True,
        old_r: Range | None = None,
        new_r: Range | None = None,
    ) -> Mtx:
        m_np = np.asanyarray(M, dtype=np.float32)

        if use_z_score:
            self.logger.debug(f"[process] Mode: Z-Score. Shape={m_np.shape}")
            return self.z_score_normalization(m_np)

        o_r = old_r if old_r is not None else (0.0, 255.0)
        n_r = new_r if new_r is not None else (0.0, 1.0)

        self.logger.debug(f"[process] Mode: Min-Max. Range={n_r}")
        result = self.normalize(m_np, o_r, n_r)

        r_min, r_max = float(np.min(result)), float(np.max(result))
        if r_min < n_r[0] or r_max > n_r[1]:
            self.logger.warning(
                f"[process] Out of bounds: [{r_min:.2f}, {r_max:.2f}] vs {n_r}"
            )

        return result
