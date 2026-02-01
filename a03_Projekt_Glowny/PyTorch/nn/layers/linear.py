from .base import LayerProtocol, Mtx


class LinearLayer(LayerProtocol):
    def __init__(self, in_features: int, out_features: int):
        self.weights: Mtx = None  # Do uzupełnienia: np.random...
        self.bias: Mtx = None  # Do uzupełnienia: np.zeros...
        self._cache_x: Mtx = None

    def forward(self, x: Mtx) -> Mtx:
        """Implementacja: x @ weights + bias"""
        ...

    def backward(self, grad: Mtx) -> Mtx:
        """Implementacja: obliczenie gradientów dla wag i wejścia"""
        ...
