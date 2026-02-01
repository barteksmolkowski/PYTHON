from .base import LayerProtocol, Mtx


class FlattenLayer(LayerProtocol):
    def __init__(self):
        self._original_shape: tuple = None

    def forward(self, x: Mtx) -> Mtx:
        """Zmiana kształtu na (batch, -1)"""
        ...

    def backward(self, grad: Mtx) -> Mtx:
        """Przywrócenie oryginalnego kształtu gradientu"""
        ...
