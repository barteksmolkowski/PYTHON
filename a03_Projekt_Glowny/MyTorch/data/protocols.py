from typing import List, Literal, Protocol, Tuple, overload

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


class BatchProcessingProtocol(Protocol):
    def create_batches(
        self, data: T4D, batch_size: int = 1, shuffle: bool = True, seed: int = 0
    ) -> List[T4D]: ...

    def process_batch(self, paths: List[FilePath]) -> T4D: ...


class CacheManagerProtocol(Protocol):
    def cache(self, data: BatchData, path: FilePath) -> bool: ...

    def load(self, path: FilePath) -> BatchData: ...


class DatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Tuple[Sample, Label]: ...


class DataDownloaderProtocol(Protocol):
    def fetch_data(self, source: str, force_reload: bool = False) -> DataResult: ...

    def save_to_disk(
        self, data: JsonData, filename: FilePath = "data.json"
    ) -> bool: ...


class DataProcessorProtocol(Protocol):
    def process_batch(
        self, items: List[RawImage], grayscale: bool = True
    ) -> BatchData: ...


class ProjectManagerProtocol(Protocol):
    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[True]) -> str: ...

    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[False]) -> str: ...

    def run_pipeline(self, source_url: str, fast_mode: bool = False) -> str: ...
