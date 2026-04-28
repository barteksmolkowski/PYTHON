import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Protocol

from MyTorch import (
    T1D,
    ImageGray,
)


class EdgeDetectorProtocol(Protocol):
    def apply(self, matrix: ImageGray) -> ImageGray: ...


class FeatureExtractorProtocol(Protocol):
    def extract_edges(self, matrix: ImageGray) -> ImageGray: ...

    def extract_features(self, matrix: ImageGray) -> List[float]: ...


class HOGProtocol(Protocol):
    def extract(self, matrix: ImageGray) -> T1D: ...


@dataclass(frozen=True, slots=True)
class EdgeDetectorABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def apply(self, matrix: ImageGray) -> ImageGray:
        pass


@dataclass(frozen=True, slots=True)
class FeatureExtractionABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def extract_edges(self, matrix: ImageGray) -> ImageGray:
        pass

    @abstractmethod
    def extract_features(self, matrix: ImageGray) -> List[float]:
        pass


@dataclass(frozen=True, slots=True)
class HOGABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def extract(self, matrix: ImageGray) -> T1D:
        pass
