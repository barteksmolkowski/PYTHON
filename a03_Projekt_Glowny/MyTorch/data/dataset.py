from dataclasses import dataclass, field
from typing import Tuple, cast

from MyTorch import BatchData, Label, LabelsMtx, Sample

from .base import DatasetABC


def validate_dataset_integrity_logic(data: BatchData, labels: LabelsMtx) -> bool:
    if not data or labels.size == 0:
        return False
    if len(data) != labels.shape[0]:
        return False
    return True


def get_dataset_item_logic(
    data: BatchData, labels: LabelsMtx, index: int = 0
) -> Tuple[Sample, Label]:
    raw_sample = data[index]
    label_item = int(labels[index])

    sample_item = cast(Sample, (raw_sample, label_item))

    return sample_item, label_item


@dataclass(frozen=True, slots=True)
class Dataset(DatasetABC):
    data: BatchData = field(repr=False)
    labels: LabelsMtx = field(repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not validate_dataset_integrity_logic(self.data, self.labels):
            error_msg = f"Integrity check failed: data_len={len(self.data)}, labels_len={len(self.labels)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Sample, Label]:
        try:
            return get_dataset_item_logic(self.data, self.labels, index)
        except IndexError:
            self.logger.error(f"Index {index} out of bounds (size: {len(self)})")
            raise
