import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Tuple, Union, cast

import numpy as np
from common_utils import class_autologger, silent

from .types import (
    BatchData,
    LabelsMtx,
    Sample,
)


class DatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Union[np.ndarray, BatchData, Sample]: ...


def validate_dataset_integrity_logic(data: BatchData, labels: LabelsMtx) -> bool:
    if not data or not labels:
        return False
    if len(data) != len(labels):
        return False
    if len(data) == 0:
        return False
    return True


def get_dataset_item_logic(
    data: BatchData, labels: LabelsMtx, index: int
) -> Tuple[Sample, Any]:
    sample_item = cast(Sample, data[index])
    label_item = labels[index]

    return sample_item, label_item


@class_autologger
@dataclass
class Dataset:
    data: BatchData
    labels: LabelsMtx

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        if not validate_dataset_integrity_logic(self.data, self.labels):
            self.logger.error(
                f"Integrity check failed: data_len={len(self.data)}, labels_len={len(self.labels)}"
            )
            raise ValueError("Dataset integrity mismatch or empty data provided.")

    @silent
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[Tuple[Sample, Any], "Dataset"]:
        if isinstance(index, slice):
            return Dataset(data=self.data[index], labels=self.labels[index])

        try:
            return get_dataset_item_logic(self.data, self.labels, index)
        except IndexError:
            self.logger.error(
                f"[Dataset] Index {index} is out of bounds (size: {len(self)})"
            )
            raise
