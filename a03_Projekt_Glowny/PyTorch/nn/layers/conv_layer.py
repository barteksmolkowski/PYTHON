from .base import LayerProtocol, Mtx


class Conv2DLayer(LayerProtocol):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ):
        self.kernels: Mtx = None
        self.biases: Mtx = None
        self._input_cache: Mtx = None

    def forward(self, x: Mtx) -> Mtx:
        """Logika splotu 2D"""
        ...

    def backward(self, grad: Mtx) -> Mtx:
        """Gradienty splotu"""
        ...
