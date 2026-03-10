from typing import Protocol

from nn import Mtx


class LayerProtocol(Protocol):
    def forward(self, x: Mtx) -> Mtx: ...

    def backward(self, grad: Mtx) -> Mtx: ...
