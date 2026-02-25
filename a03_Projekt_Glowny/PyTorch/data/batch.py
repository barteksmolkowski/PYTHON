import random
from typing import Protocol, TypeAlias
from preprocessing.pipeline import ImageDataPreprocessing
import numpy as np
import logging

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class BatchProcessingProtocol(Protocol):
    def create_batches(
        self, data: MtxList, batch_size: int, shuffle: bool = True
    ) -> list[MtxList]: ...

    def process_batch(self, paths: list[str]) -> MtxList: ...


class BatchProcessing:
    logger: logging.Logger

    def __init__(self, image_data_preprocessing=None) -> None:
        if image_data_preprocessing is None:
            self.logger.debug(
                "[__init__] image_data_preprocessing is None, using default ImageDataPreprocessing"
            )
        self.image_data_preprocessing = (
            image_data_preprocessing or ImageDataPreprocessing()
        )

    def create_batches(
        self, data: MtxList, batch_size: int, shuffle: bool
    ) -> list[MtxList]:
        data_copy = list(data)
        if shuffle:
            random.shuffle(data_copy)

        num_batches = (len(data_copy) + batch_size - 1) // batch_size
        batches = np.array_split(data_copy, num_batches)

        return [list(b) for b in batches]

    def process_batch(self, paths: list[str]) -> MtxList:
        all_samples: MtxList = []

        for path in paths:
            image_results = self.image_data_preprocessing.preprocess(path=path)

            if image_results:

                for channel_samples in image_results:

                    all_samples.extend(channel_samples)

        self.logger.info(f"[process_batch] Total samples in batch: {len(all_samples)}")
        return all_samples
