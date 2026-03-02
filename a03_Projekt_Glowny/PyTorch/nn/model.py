import logging
import sys
from typing import List, Optional

from common_utils import class_autologger
from nn import LayerProtocol, Mtx


class Sequential:
    def __init__(self, layers: Optional[List[LayerProtocol]] = None):
        self.layers: List[LayerProtocol] = layers or []

    def add(self, layer: LayerProtocol) -> None: ...

    def forward(self, x: Mtx) -> Mtx: ...

    def backward(self, grad: Mtx) -> Mtx: ...


@class_autologger
class NeuralNetwork:
    logger: logging.Logger

    def __init__(self, model: Optional["Sequential"] = None):
        if model is None:
            self.logger.critical(
                "[NeuralNetwork] Failed to initialize: Model is None. Execution halted."
            )
            print(
                "\n[CRITICAL ERROR] NeuralNetwork requires a Sequential model to function."
            )
            sys.exit(1)

        self.model = model

    def predict(self, x: Mtx) -> Mtx: ...

    def save(self, path: Optional[str] = None) -> None:
        """save zrobić w .npz"""
        ...

    def load(self, path: Optional[str] = None) -> None:
        """load zrobić w .npz"""
        ...
