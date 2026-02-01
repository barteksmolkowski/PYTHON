from typing import List

from .layers.base import LayerProtocol
from .tensor import Mtx


class Sequential:
    def __init__(self, layers: List[LayerProtocol] = None):
        self.layers: List[LayerProtocol] = layers or []

    def add(self, layer: LayerProtocol) -> None: ...

    def forward(self, x: Mtx) -> Mtx: ...

    def backward(self, grad: Mtx) -> Mtx: ...


class NeuralNetwork:
    def __init__(self, model: Sequential):
        self.model = model

    def predict(self, x: Mtx) -> Mtx: ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...
