from typing import Literal, Protocol, Union, overload

import numpy as np

Mtx = np.ndarray
MtxList = list[np.ndarray]


class DataDownloaderProtocol(Protocol):
    def fetch_data(
        self, source: str, force_reload: bool = False
    ) -> Union[list, dict]: ...
    def save_to_disk(self, data: Union[list, dict]) -> bool: ...


class DataProcessorProtocol(Protocol):
    def process_batch(self, items: list, grayscale: bool = True) -> bool: ...


class ProjectManagerProtocol(Protocol):
    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[True]) -> bool: ...

    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[False]) -> str: ...

    def run_pipeline(
        self, source_url: str, fast_mode: bool = False
    ) -> Union[str, bool]: ...


class DataDownloader:
    def fetch_data(self, source: str, force_reload: bool = False) -> Union[list, dict]:
        return []

    def save_to_disk(self, data: Union[list, dict]) -> bool:
        return True


class DataProcessor:
    def process_batch(self, items: list, grayscale: bool = True) -> bool:
        return True


class ProjectManager:
    def __init__(self):
        self.downloader: DataDownloaderProtocol = DataDownloader()
        self.processor: DataProcessorProtocol = DataProcessor()

    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[True]) -> bool: ...

    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[False]) -> str: ...

    def run_pipeline(
        self, source_url: str, fast_mode: bool = False
    ) -> Union[str, bool]:
        ...
