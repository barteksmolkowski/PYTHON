from typing import Protocol

import numpy as np

Mtx = np.ndarray


class LayerProtocol(Protocol):
    def forward(self, x: Mtx) -> Mtx: ...

    def backward(self, grad: Mtx) -> Mtx: ...
