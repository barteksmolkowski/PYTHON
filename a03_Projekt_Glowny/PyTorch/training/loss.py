from typing import Protocol, TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray


class LossProtocol(Protocol):
    def calculate(self, y_pred: Mtx, y_true: Mtx) -> float: ...
    def derivative(self, y_pred: Mtx, y_true: Mtx) -> Mtx: ...


class MSE:
    def calculate(self, y_pred: Mtx, y_true: Mtx) -> float:
        pass

    def derivative(self, y_pred: Mtx, y_true: Mtx) -> Mtx:
        pass


class CrossEntropy:
    def calculate(self, y_pred: Mtx, y_true: Mtx) -> float:
        pass

    def derivative(self, y_pred: Mtx, y_true: Mtx) -> Mtx:
        pass
