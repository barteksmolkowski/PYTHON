from dataclasses import dataclass, field
from typing import List

import numpy as np

from data.base import BatchProcessingABC
from MyTorch import T4D, FilePath
from preprocessing import ImageDataPreprocessing, ImageDataPreprocessingProtocol


def create_batches_logic(
    data: T4D,
    batch_size: int = 1,
    shuffle: bool = True,
    seed: int = 0,
) -> List[T4D]:
    if data.size == 0:
        return []

    rng = np.random.default_rng(seed)
    data_copy = rng.permutation(data) if shuffle else np.copy(data)

    if batch_size <= 0:
        return [data_copy]

    num_samples = len(data_copy)
    indices = np.arange(batch_size, num_samples, batch_size)
    return np.array_split(data_copy, indices) if num_samples > 0 else []


def process_single_path_logic(
    path: FilePath, preprocessor: ImageDataPreprocessingProtocol
) -> T4D:
    image_results = preprocessor.preprocess(path=path)

    if not image_results:
        return np.zeros((0, 0, 0, 0), dtype=np.float32)

    return np.concatenate(image_results, axis=0)


@dataclass(frozen=True, slots=True)
class BatchProcessing(BatchProcessingABC):
    preprocessor: ImageDataPreprocessingProtocol = field(
        default_factory=ImageDataPreprocessing
    )

    def create_batches(
        self, data: T4D, batch_size: int = 1, shuffle: bool = True, seed: int = 0
    ) -> List[T4D]:
        return create_batches_logic(data, batch_size, shuffle, seed)

    def process_batch(self, paths: List[FilePath]) -> T4D:
        all_samples: List[T4D] = []
        for path in paths:
            try:
                samples = process_single_path_logic(path, self.preprocessor)
                all_samples.append(samples)
            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}")
                continue

        if not all_samples:
            return np.array([], dtype=float).reshape(0, 0, 0, 0)

        return np.concatenate(all_samples, axis=0)
