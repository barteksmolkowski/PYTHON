import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

from MyTorch import T2D


class LossProtocol(Protocol):
    def calculate(self, y_pred: T2D, y_true: T2D) -> float: ...
    def derivative(self, y_pred: T2D, y_true: T2D) -> T2D: ...


class OptimizerProtocol(Protocol):
    def update(self, weights: T2D, grad: T2D) -> T2D: ...


class TrainerProtocol(Protocol):
    def train(self, epochs: int) -> None: ...

    def evaluate(self) -> float: ...


@dataclass(frozen=True, slots=True)
class LossABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def calculate(self, y_pred: T2D, y_true: T2D) -> float: ...

    @abstractmethod
    def derivative(self, y_pred: T2D, y_true: T2D) -> T2D: ...


@dataclass(frozen=True, slots=True)
class OptimizerABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def update(self, weights: T2D, grad: T2D) -> T2D: ...


@dataclass(frozen=False, slots=True)
class TrainerABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def train(self, epochs: int) -> None: ...

    @abstractmethod
    def evaluate(self) -> float: ...
