from typing import Any

import numpy as np
from a03_Projekt_Glowny.MyTorch.nn.layers.interfaces import LayerProtocol

from common_utils import class_autologger

Mtx = np.ndarray


@class_autologger
class ReLULayer(LayerProtocol):
    def __init__(self):
        self._input_mask: Mtx = None

    def forward(self, x: Mtx) -> Mtx:
        self._input_mask = x > 0

        if np.all(x <= 0):
            self.logger.debug(
                f"[forward] All input values are <= 0; ReLU will output zeros"
            )

        output = x * self._input_mask
        return output

    def backward(self, grad: Mtx) -> Mtx:
        if self._input_mask is None:
            self.logger.error(
                "[backward] Forward pass must be called before backward; mask is missing"
            )
            return np.zeros_like(grad)

        dx = grad * self._input_mask
        self.logger.debug(
            f"[backward] Gradient propagated through mask with shape {dx.shape}"
        )
        return dx


@class_autologger
class SigmoidLayer(LayerProtocol):
    def __init__(self):
        self._last_output: Mtx = None

    def forward(self, x: Mtx) -> Mtx:
        if np.any(x < -709) or np.any(x > 709):
            self.logger.debug(
                f"[forward] Extreme input values detected (x < -709 or x > 709); exp(x) might overflow"
            )

        output = 1 / (1 + np.exp(-x))
        self._last_output = output
        return output

    def backward(self, grad: Mtx) -> Mtx:
        if self._last_output is None:
            self.logger.error(
                "[backward] Forward pass must be called before backward; last_output is missing"
            )
            return np.zeros_like(grad)

        ds = self._last_output * (1 - self._last_output)
        dx = grad * ds

        self.logger.debug(
            f"[backward] Computed sigmoid gradient for output with mean value {np.mean(self._last_output):.4f}"
        )
        return dx
