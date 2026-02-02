from typing import List, Optional

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

    def save(self, path: Optional[str] = None) -> None:
        """save zrobić w .npz"""
        ...

    def load(self, path: Optional[str] = None) -> None:
        """load zrobić w .npz"""
        ...
