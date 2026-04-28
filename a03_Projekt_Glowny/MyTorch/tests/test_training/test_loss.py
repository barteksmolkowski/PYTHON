import numpy as np

Mtx = np.ndarray
from common_utils import class_autologger


@class_autologger
class MSE:
    def calculate(self, y_pred: Mtx, y_true: Mtx) -> float:
        if y_pred.shape != y_true.shape:
            self.logger.error(
                f"[calculate] Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}"
            )
            return 0.0

        loss = np.mean((y_pred - y_true) ** 2)
        self.logger.debug(f"[calculate] Computed MSE loss value: {loss}")
        return float(loss)

    def derivative(self, y_pred: Mtx, y_true: Mtx) -> Mtx:
        if y_pred.shape != y_true.shape:
            self.logger.error(
                f"[derivative] Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}"
            )
            return np.zeros_like(y_pred)

        grad = 2 * (y_pred - y_true) / y_pred.size
        self.logger.debug(
            f"[derivative] Gradient calculated for input size: {y_pred.size}"
        )
        return grad


@class_autologger
class CrossEntropy:
    def calculate(self, y_pred: Mtx, y_true: Mtx) -> float:
        if y_pred.shape != y_true.shape:
            self.logger.error(
                f"[calculate] Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}"
            )
            return 0.0

        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        self.logger.debug(
            f"[calculate] Values clipped using epsilon: {epsilon} to prevent log(0)"
        )

        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return float(loss)

    def derivative(self, y_pred: Mtx, y_true: Mtx) -> Mtx:
        if y_pred.shape != y_true.shape:
            self.logger.error(
                f"[derivative] Shape mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}"
            )
            return np.zeros_like(y_pred)

        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        grad = (y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped))
        grad /= y_pred.size

        self.logger.debug(
            f"[derivative] Gradient computed for CrossEntropy with clipped y_pred"
        )
        return grad
