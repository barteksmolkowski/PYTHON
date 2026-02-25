from typing import Any, Literal, Protocol, TypeAlias, overload
import logging
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


@class_autologger
class Normalization:
    logger: logging.Logger

    def normalize(self, M: Mtx, old_r: Range = (0, 255), new_r: Range = (0, 1)) -> Mtx:
        denominator = old_r[1] - old_r[0]
        if denominator == 0:
            self.logger.error(
                f"[normalize] Zero denominator: old_r={old_r}. Data loss: returning constant matrix value={new_r[0]}"
            )
            return np.zeros_like(M, dtype=np.float32) + new_r[0]

        self.logger.debug(
            f"[normalize] Scaling: old_range={old_r} -> new_range={new_r}, denominator={denominator}"
        )
        return (M - old_r[0]) * (new_r[1] - new_r[0]) / denominator + new_r[0]

    def z_score_normalization(self, M: Mtx) -> Mtx:
        eps = 1e-8
        mean_val, std_val = np.mean(M), np.std(M)

        if std_val < eps:
            self.logger.warning(
                f"[z_score_normalization] Low variance detected: std={std_val:.8f}. Result may be unstable."
            )

        self.logger.debug(
            f"[z_score_normalization] Calculated stats: mean={mean_val:.4f}, std={std_val:.4f}, eps={eps}"
        )
        return (M - mean_val) / (std_val + eps)

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
        M_np = np.asanyarray(M, dtype=np.float32)

        if use_z_score:
            self.logger.debug(
                f"[process] Logic path: Z-Score. Input shape={M_np.shape}"
            )
            result = self.z_score_normalization(M_np)
            self.logger.info(
                f"[process] Validated 1 items. Z-Score normalization complete."
            )
            return result

        o_r = old_r or (0.0, 255.0)
        n_r = new_r or (0.0, 1.0)

        self.logger.debug(
            f"[process] Logic path: Min-Max. old_range={o_r}, new_range={n_r}"
        )
        result = self.normalize(M_np, o_r, n_r)

        if np.min(result) < n_r[0] or np.max(result) > n_r[1]:
            self.logger.warning(
                f"[process] Out of bounds: result range [{np.min(result):.2f}, {np.max(result):.2f}] "
                f"exceeds target range {n_r}"
            )

        self.logger.info(
            f"[process] Validated 1 items. Min-Max normalization complete."
        )
        return result
