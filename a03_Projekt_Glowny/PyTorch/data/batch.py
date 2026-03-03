import logging
import random
from typing import List, Optional, Protocol

import numpy as np
from common_utils import autologger
from preprocessing import ImageDataPreprocessing

from .types import BatchData, FilePath, OptBatchData


class BatchProcessingProtocol(Protocol):
    def create_batches(
        self,
        data: BatchData,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> List[BatchData]: ...

    def process_batch(self, paths: List[FilePath]) -> OptBatchData: ...


@autologger
class BatchProcessing:
    logger: logging.Logger

    def __init__(
        self, image_data_preprocessing: Optional[ImageDataPreprocessing] = None
    ) -> None:
        if image_data_preprocessing is None:
            self.logger.debug(
                "[__init__] preprocessor is None, using default ImageDataPreprocessing"
            )
        self.image_data_preprocessing = (
            image_data_preprocessing or ImageDataPreprocessing()
        )

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

        data_copy = list(data)
        if shuffle:
            current_seed = seed if seed is not None else random.randint(0, 1000000)
            self.logger.info(f"[create_batches] Shuffling with seed: {current_seed}")
            random.seed(current_seed)
            random.shuffle(data_copy)

        num_batches = (len(data_copy) + batch_size - 1) // batch_size
        batches = np.array_split(data_copy, num_batches)

        return [list(b) for b in batches]

    def process_batch(self, paths: List[FilePath]) -> OptBatchData:
        if not paths:
            self.logger.error("[process_batch] Path list is None or empty.")
            return None

        all_samples: BatchData = []

        for path in paths:
            try:
                image_results = self.image_data_preprocessing.preprocess(path=path)
                if image_results:
                    for channel_samples in image_results:
                        all_samples.extend(channel_samples)
            except Exception as e:
                self.logger.error(f"[process_batch] Critical file error: {path} | {e}")
                continue

        if not all_samples:
            self.logger.warning(
                "[process_batch] No samples were successfully processed."
            )
            return None

        self.logger.info(f"[process_batch] Processed {len(all_samples)} samples.")
        return all_samples
