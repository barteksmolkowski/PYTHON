import logging
from typing import Protocol, Union

import numpy as np
from common_utils import autologger

from .types import (
    BatchData,
    LabelsMtx,
    OptBatchData,
    OptLabelsMtx,
    Sample,
)


class DatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Union[np.ndarray, BatchData, Sample]: ...


@autologger
class Dataset:
    logger: logging.Logger

    def __init__(self, data: OptBatchData, labels: OptLabelsMtx) -> None:
        if data is None or labels is None:
            self.logger.error("[Dataset] Failed to initialize: data or labels is None.")
            raise ValueError("Dataset requires both data and labels to be provided.")

        assert len(data) > 0, "[Dataset] Data list is empty."
        assert len(data) == len(labels), (
            f"[Dataset] Mismatch: {len(data)} images vs {len(labels)} labels."
        )

        self.data: BatchData = data
        self.labels: LabelsMtx = labels
        self.logger.info(f"[Dataset] Successfully loaded {len(self)} samples.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Union[int, slice]) -> Union[Sample, "Dataset"]:
        if isinstance(index, slice):
            return Dataset(data=self.data[index], labels=self.labels[index])

        try:
            return (self.data[index], self.labels[index])
        except IndexError:
            self.logger.error(f"[Dataset] Out of bounds: {index}")
            raise
