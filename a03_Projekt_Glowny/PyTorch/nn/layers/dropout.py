from .base import LayerProtocol, Mtx


class DropoutLayer(LayerProtocol):
    def __init__(self, probability: float = 0.5):
        self.probability = probability
        self._mask: Mtx = None

    def forward(self, x: Mtx) -> Mtx: ...

    def backward(self, grad: Mtx) -> Mtx: ...
