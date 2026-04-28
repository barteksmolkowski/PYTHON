import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from MyTorch import T, apply_to_methods


class LayerProtocol(Protocol):
    def forward(self, x: T) -> T: ...

    def backward(self, grad: T) -> T: ...


@apply_to_methods(decorators=abstractmethod, method_names=[])
@dataclass(frozen=True, slots=True)
class LayerABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def forward(self, x: T) -> T:
        return np.array([], dtype=float).reshape(0, 2)

    @abstractmethod
    def backward(self, grad: T) -> T:
        return np.array([], dtype=float).reshape(0, 2)


@dataclass(frozen=True, slots=True)
class ActivationABC(LayerABC):
    _last_output: T = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=float)
    )
