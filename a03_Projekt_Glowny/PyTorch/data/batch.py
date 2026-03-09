import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np
from common_utils import class_autologger
from preprocessing import ImageDataPreprocessing, ImageDataPreprocessingProtocol

from .types import (
    BatchData,
    FilePath,
    OptBatchData,
)


def create_batches_logic(
    data: BatchData,
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[BatchData]:
    if not data:
        return []

    data_copy = list(data)
    if shuffle:
        current_seed = seed if seed is not None else random.randint(0, 1000000)
        random.seed(current_seed)
        random.shuffle(data_copy)

    if batch_size <= 0:
        return [data_copy]

    num_batches = (len(data_copy) + batch_size - 1) // batch_size
    batches = np.array_split(data_copy, num_batches)
    return [list(b) for b in batches]


def process_single_path_logic(
    path: FilePath, preprocessor: ImageDataPreprocessingProtocol
) -> BatchData:
    samples: BatchData = []
    image_results = preprocessor.preprocess(path=path)

    if image_results:
        for channel_samples in image_results:
            samples.extend(channel_samples)
    return samples


class BatchProcessingProtocol(Protocol):
    def create_batches(
        self,
        data: BatchData,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> List[BatchData]: ...

    def process_batch(self, paths: List[FilePath]) -> OptBatchData: ...


@dataclass
@class_autologger
class BatchProcessing:
    preprocessor: ImageDataPreprocessingProtocol = field(
        default_factory=ImageDataPreprocessing
    )

    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        assert self.preprocessor is not None
        self.logger.debug("BatchProcessing initialized with stateless engines.")

    def create_batches(
        self,
        data: BatchData,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> List[BatchData]:
        if not data:
            self.logger.warning("[create_batches] Input data is empty.")
            return []

        if shuffle:
            self.logger.info(
                f"[create_batches] Shuffling data (seed provided: {seed is not None})"
            )

        return create_batches_logic(data, batch_size, shuffle, seed)

    def process_batch(self, paths: List[FilePath]) -> OptBatchData:
        if not paths:
            self.logger.error("[process_batch] Path list is empty.")
            return None

        all_samples: BatchData = []

        for path in paths:
            try:
                samples = process_single_path_logic(path, self.preprocessor)
                all_samples.extend(samples)
            except Exception as e:
                self.logger.error(f"[process_batch] Error processing {path}: {e}")
                continue

        if not all_samples:
            self.logger.warning("[process_batch] No samples successfully processed.")
            return None

        self.logger.info(f"[process_batch] Total processed samples: {len(all_samples)}")
        return all_samples
