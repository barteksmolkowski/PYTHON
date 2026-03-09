import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Protocol, Union, overload

import numpy as np
import requests
from common_utils import class_autologger

from .types import (
    DataResult,
    FilePath,
    JsonData,
    ProcessedImage,
    is_color,
    is_grayscale,
)


class DataDownloaderProtocol(Protocol):
    def fetch_data(self, source: str, force_reload: bool = False) -> DataResult: ...

    def save_to_disk(self, data: JsonData, filename: FilePath) -> bool: ...


class DataProcessorProtocol(Protocol):
    def process_batch(
        self, items: List[Any], grayscale: bool = True
    ) -> List[ProcessedImage]: ...


class ProjectManagerProtocol(Protocol):
    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[True]) -> bool: ...

    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[False]) -> str: ...

    def run_pipeline(
        self, source_url: str, fast_mode: bool = False
    ) -> Union[str, bool]: ...


def save_json_to_disk_logic(data: JsonData, filename: FilePath) -> bool:
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        return True
    except (IOError, TypeError):
        return False


def fetch_remote_data_logic(url: str, timeout: int = 10) -> DataResult:
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                return response.content
        return {}
    except requests.exceptions.RequestException:
        return {}


def load_local_json_logic(path: FilePath) -> DataResult:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (IOError, json.JSONDecodeError):
            return {}
    return {}


def transform_image_to_normalized_logic(
    item: Any, grayscale: bool = True
) -> Optional[ProcessedImage]:
    img = np.array(item)

    if grayscale and is_color(img):
        pass

    if grayscale and not is_grayscale(img):
        return None

    return img.astype(np.float32) / 255.0


def batch_transform_engine(
    items: List[Any], grayscale: bool = True
) -> List[ProcessedImage]:
    processed = []
    for item in items:
        res = transform_image_to_normalized_logic(item, grayscale)
        if res is not None:
            processed.append(res)
    return processed


def determine_pipeline_result_logic(
    success_count: int, fast_mode: bool
) -> Union[str, bool]:
    if success_count == 0:
        return False if fast_mode else "Failed: No data or processing failed"

    success_msg = f"Pipeline finished. Processed {success_count} items."
    return True if fast_mode else success_msg


def _prepare_items_for_processing(raw_data: DataResult) -> List[Any]:
    if not raw_data:
        return []
    return raw_data if isinstance(raw_data, list) else [raw_data]


@class_autologger
@dataclass
class DataDownloader:
    timeout: int = 10

    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"DataDownloader initialized (timeout={self.timeout}s).")

    def save_to_disk(self, data: JsonData, filename: FilePath = "data.json") -> bool:
        success = save_json_to_disk_logic(data, filename)

        if not success:
            self.logger.error(f"[save_to_disk] Failed to save JSON to: {filename}")
        else:
            self.logger.info(f"[save_to_disk] Successfully saved data to: {filename}")

        return success

    def fetch_data(self, source: str, force_reload: bool = False) -> DataResult:
        is_url = source.startswith(("http://", "https://"))

        if is_url or force_reload:
            self.logger.info(f"[fetch_data] Fetching remote resource: {source}")
            result = fetch_remote_data_logic(source, timeout=self.timeout)

            if not result:
                self.logger.error(
                    f"[fetch_data] Remote fetch failed or returned empty: {source}"
                )
            return result

        self.logger.debug(f"[fetch_data] Loading local file: {source}")
        result = load_local_json_logic(source)

        if not result:
            if not os.path.exists(source):
                self.logger.warning(f"[fetch_data] File not found: {source}")
            else:
                self.logger.error(f"[fetch_data] Failed to parse local file: {source}")

        return result


@class_autologger
@dataclass
class DataProcessor:
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            "DataProcessor initialized with stateless normalization engines."
        )

    def process_batch(
        self, items: List[Any], grayscale: bool = True
    ) -> List[ProcessedImage]:
        if not items:
            self.logger.warning("[process_batch] Input items list is empty.")
            return []

        self.logger.debug(
            f"[process_batch] Processing {len(items)} items (grayscale={grayscale})"
        )

        processed_items = batch_transform_engine(items, grayscale)

        success_count = len(processed_items)
        total_count = len(items)

        if success_count < total_count:
            skipped = total_count - success_count
            self.logger.warning(f"[process_batch] Skipped {skipped} invalid samples.")

        self.logger.info(
            f"[process_batch] Successfully processed {success_count}/{total_count} samples."
        )

        return processed_items


@class_autologger
@dataclass
class ProjectManager:
    downloader: DataDownloaderProtocol = field(default_factory=DataDownloader)
    processor: DataProcessorProtocol = field(default_factory=DataProcessor)

    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("ProjectManager initialized as High-Level Orchestrator.")

        assert self.downloader is not None
        assert self.processor is not None

    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[True]) -> bool: ...

    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[False]) -> str: ...

    def run_pipeline(
        self, source_url: str, fast_mode: bool = False
    ) -> Union[str, bool]:
        self.logger.info(
            f"[run_pipeline] Starting pipeline | Source: {source_url} | FastMode: {fast_mode}"
        )

        raw_data = self.downloader.fetch_data(source_url)

        items_to_process = _prepare_items_for_processing(raw_data)

        if not items_to_process:
            self.logger.error("[run_pipeline] Pipeline aborted: No items to process.")
            return determine_pipeline_result_logic(0, fast_mode)

        processed_data = self.processor.process_batch(items_to_process)

        result = determine_pipeline_result_logic(len(processed_data), fast_mode)

        self.logger.info(f"[run_pipeline] Execution finished. Result: {result}")
        return result
