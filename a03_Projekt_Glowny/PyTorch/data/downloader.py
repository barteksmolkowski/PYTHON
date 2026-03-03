import json
import logging
import os
from typing import Any, List, Literal, Protocol, Union, overload

import numpy as np
import requests
from common_utils import autologger

from . import DataResult, FilePath, JsonData


class DataDownloaderProtocol(Protocol):
    def fetch_data(self, source: str, force_reload: bool = False) -> DataResult: ...

    def save_to_disk(self, data: JsonData, filename: FilePath) -> bool: ...


class DataProcessorProtocol(Protocol):
    def process_batch(self, items: List[Any], grayscale: bool = True) -> bool: ...


class ProjectManagerProtocol(Protocol):
    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[True]) -> bool: ...

    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[False]) -> str: ...

    def run_pipeline(
        self, source_url: str, fast_mode: bool = False
    ) -> Union[str, bool]: ...


@autologger
class DataDownloader:
    logger: logging.Logger

    def fetch_data(self, source: str, force_reload: bool = False) -> DataResult:
        is_url = source.startswith(("http://", "https://"))

        if is_url or force_reload:
            try:
                response = requests.get(source, timeout=10)
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError:
                        return response.content

                self.logger.error(f"HTTP Error: {response.status_code}")
                return {}
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Connection failed: {e}")
                return {}

        if os.path.exists(source):
            with open(source, "r", encoding="utf-8") as file:
                return json.load(file)

        self.logger.warning(f"File not found: {source}")
        return {}

    def save_to_disk(self, data: JsonData, filename: FilePath = "data.json") -> bool:
        try:
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)
            return True
        except (IOError, TypeError) as e:
            self.logger.error(f"Save failed: {e}")
            return False


class DataProcessor:
    def process_batch(self, items: List[Any], grayscale: bool = True) -> bool:
        return True


@autologger
class ProjectManager:
    logger: logging.Logger

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
        self.logger.info(
            f"Running pipeline in {'fast' if fast_mode else 'standard'} mode."
        )
        return True if fast_mode else "Pipeline Completed"
