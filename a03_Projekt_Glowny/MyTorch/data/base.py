import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

from MyTorch import (
    T4D,
    BatchData,
    DataResult,
    FilePath,
    JsonData,
    Label,
    RawImage,
    Sample,
)


@dataclass(frozen=True, slots=True)
class BatchProcessingABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def create_batches(
        self, data: T4D, batch_size: int = 1, shuffle: bool = True, seed: int = 0
    ) -> List[T4D]:
        pass

    @abstractmethod
    def process_batch(self, paths: List[FilePath]) -> T4D:
        pass


@dataclass(frozen=True, slots=True)
class CacheManagerABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def load(self, path: FilePath) -> BatchData:
        pass

    @abstractmethod
    def cache(self, data: BatchData, path: FilePath) -> bool:
        pass


@dataclass(frozen=True, slots=True)
class DatasetABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Sample, Label]:
        pass


@dataclass(frozen=True, slots=True)
class DataDownloaderABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def fetch_data(self, source: str, force_reload: bool = False) -> DataResult:
        pass

    @abstractmethod
    def save_to_disk(self, data: JsonData, filename: FilePath = "data.json") -> bool:
        pass


@dataclass(frozen=True, slots=True)
class DataProcessorABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def process_batch(self, items: List[RawImage], grayscale: bool = True) -> BatchData:
        pass


@dataclass(frozen=True, slots=True)
class ProjectManagerABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def run_pipeline(self, source_url: str, fast_mode: bool = False) -> str:
        pass
