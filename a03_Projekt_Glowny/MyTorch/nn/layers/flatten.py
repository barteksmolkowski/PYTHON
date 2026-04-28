from dataclasses import dataclass, field

import numpy as np
from interfaces import LayerABC

from MyTorch import T2D, T4D, Shape


def flatten_fwd(x: T4D) -> tuple[T2D, Shape]:
    original_shape = x.shape
    batch_size = x.shape[0]
    result = x.reshape(batch_size, -1).astype(np.float32)
    return result, original_shape


def flatten_bwd(grad: T2D, original_shape: Shape) -> T4D:
    return grad.reshape(original_shape).astype(np.float32)


@dataclass(frozen=True, slots=True)
class Flatten(LayerABC):
    _original_shape: Shape = field(init=False, repr=False, default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()

    def forward(self, x: T4D) -> T2D:
        result, shape = flatten_fwd(x)
        object.__setattr__(self, "_original_shape", shape)
        return result

    def backward(self, grad: T2D) -> T4D:
        return flatten_bwd(grad, self._original_shape)
