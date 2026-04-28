import json
import os
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import List, Literal, cast, overload

import numpy as np
import requests
from requests.exceptions import RequestException

from MyTorch import BatchData, DataResult, FilePath, ImageGray, JsonData, RawImage

from .base import (
    DataDownloaderABC,
    DataProcessorABC,
    ProjectManagerABC,
)
from .protocols import DataDownloaderProtocol, DataProcessorProtocol


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
    except (RequestException, Exception):
        return {}


def load_local_json_logic(path: FilePath) -> DataResult:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (IOError, JSONDecodeError):
        return {}


def transform_image_to_normalized_logic(
    item: RawImage, grayscale: bool = True
) -> ImageGray:
    img = np.array(item).astype(np.float32) / 255.0

    assert img.ndim == 2, f"Expected 2D ImageGray, got {img.ndim}D"

    return img


def batch_transform_engine(items: List[RawImage], grayscale: bool = True) -> BatchData:
    processed: List[ImageGray] = []
    for item in items:
        res = transform_image_to_normalized_logic(item, grayscale)
        if res.size > 0:
            processed.append(res)
    return processed


def determine_pipeline_result_logic(success_count: int, fast_mode: bool = False) -> str:
    if success_count == 0:
        return "0" if fast_mode else "Failed: No data or processing failed"

    success_msg = f"Pipeline finished. Processed {success_count} items."
    return "1" if fast_mode else success_msg


def _prepare_items_for_processing(raw_data: DataResult) -> List[DataResult]:
    if not raw_data:
        return []

    if isinstance(raw_data, list):
        return cast(List[DataResult], raw_data)

    return [raw_data]


@dataclass(frozen=True, slots=True)
class DataDownloader(DataDownloaderABC):
    timeout: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.timeout > 60:
            self.logger.warning(f"High timeout detected: {self.timeout}s")

    def fetch_data(self, source: str, force_reload: bool = False) -> DataResult:
        is_url = source.startswith(("http://", "https://"))

        if is_url or force_reload:
            result = fetch_remote_data_logic(source, timeout=self.timeout)
            if not result:
                self.logger.error(f"Remote fetch failed or returned empty: {source}")
                return {}
            return result

        if not os.path.exists(source):
            self.logger.warning(f"File not found: {source}")
            return {}

        result = load_local_json_logic(source)
        if not result:
            self.logger.error(f"Failed to parse local file: {source}")
            return {}

        return result

    def save_to_disk(self, data: JsonData, filename: FilePath = "data.json") -> bool:
        success = save_json_to_disk_logic(data, filename)
        if not success:
            self.logger.error(f"Failed to save JSON to: {filename}")
        return success


@dataclass(frozen=True, slots=True)
class DataProcessor(DataProcessorABC):
    def process_batch(self, items: List[RawImage], grayscale: bool = True) -> BatchData:
        if not items:
            self.logger.warning("Input items list is empty.")
            return []

        processed_items = batch_transform_engine(items, grayscale)

        success_count = len(processed_items)
        total_count = len(items)

        if success_count < total_count:
            self.logger.warning(
                f"Skipped {total_count - success_count} invalid samples."
            )

        self.logger.info(
            f"Successfully processed {success_count}/{total_count} samples."
        )
        return processed_items


@dataclass(frozen=True, slots=True)
class ProjectManager(ProjectManagerABC):
    downloader: DataDownloaderProtocol = field(default_factory=DataDownloader)
    processor: DataProcessorProtocol = field(default_factory=DataProcessor)

    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[True]) -> str: ...

    @overload
    def run_pipeline(self, source_url: str, fast_mode: Literal[False]) -> str: ...

    def run_pipeline(self, source_url: str, fast_mode: bool = False) -> str:
        self.logger.info(
            f"Starting pipeline | Source: {source_url} | FastMode: {fast_mode}"
        )

        raw_data = self.downloader.fetch_data(source_url)

        items_to_process = _prepare_items_for_processing(raw_data)

        valid_images: List[RawImage] = []
        for item in items_to_process:
            if isinstance(item, np.ndarray) and item.ndim == 3:
                valid_images.append(item)
            else:
                self.logger.debug(f"Skipping non-image item: {type(item).__name__}")

        if not valid_images:
            self.logger.error("Pipeline aborted: No valid 3D images (RawImage) found.")
            return determine_pipeline_result_logic(0, fast_mode)

        processed_data = self.processor.process_batch(valid_images)

        result = determine_pipeline_result_logic(len(processed_data), fast_mode)

        self.logger.info(f"Execution finished. Result: {result}")
        return result
