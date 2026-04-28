import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from MyTorch import (
    T4D,
    FilePath,
    Sequential,
)


class ModelProtocol(Protocol):
    def predict(self, x: T4D) -> T4D: ...

    def save(self, path: FilePath) -> None: ...

    def load(self, path: FilePath) -> None: ...


@dataclass(frozen=True, slots=True)
class OrchestratorABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))


@dataclass(frozen=True, slots=True)
class ModelABC(ABC):
    model: Sequential = field(default_factory=Sequential, repr=False)
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    def predict(self, x: T4D) -> T4D:
        return np.array([], dtype=np.float32).reshape(0, 0, 0, 0)

    def save(self, path: FilePath) -> None:
        pass

    def load(self, path: FilePath) -> None:
        pass
