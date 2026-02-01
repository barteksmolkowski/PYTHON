from .base import LayerProtocol, Mtx


class ReLULayer(LayerProtocol):
    def __init__(self):
        self._input_mask: Mtx = None

    def forward(self, x: Mtx) -> Mtx: ...

    def backward(self, grad: Mtx) -> Mtx: ...


class SigmoidLayer(LayerProtocol):
    def __init__(self):
        self._last_output: Mtx = None

    def forward(self, x: Mtx) -> Mtx: ...

    def backward(self, grad: Mtx) -> Mtx: ...
