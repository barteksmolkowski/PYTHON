from dataclasses import dataclass

import numpy as np
from interfaces import LossABC

from MyTorch import T2D


def compute_cross_entropy_logic(y_pred: T2D, y_true: T2D) -> float:
    return 0.0


def compute_cross_entropy_derivative_logic(y_pred: T2D, y_true: T2D) -> T2D:
    return np.array([], dtype=np.float32).reshape(0, 0)


def compute_mse_loss_logic(y_pred: T2D, y_true: T2D) -> float:
    return 0.0


def compute_mse_derivative_logic(y_pred: T2D, y_true: T2D) -> T2D:
    return np.array([], dtype=np.float32).reshape(0, 0)


@dataclass(frozen=True, slots=True)
class MSE(LossABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    def calculate(self, y_pred: T2D, y_true: T2D) -> float:
        return compute_mse_loss_logic(y_pred, y_true)

    def derivative(self, y_pred: T2D, y_true: T2D) -> T2D:
        return compute_mse_derivative_logic(y_pred, y_true)


@dataclass(frozen=True, slots=True)
class CrossEntropy(LossABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    def calculate(self, y_pred: T2D, y_true: T2D) -> float:
        return compute_cross_entropy_logic(y_pred, y_true)

    def derivative(self, y_pred: T2D, y_true: T2D) -> T2D:
        return compute_cross_entropy_derivative_logic(y_pred, y_true)
