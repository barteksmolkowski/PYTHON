import logging
from typing import TypeAlias

import numpy as np
import pytest
from common_utils import class_autologger, silent
from data.dataset import Dataset

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


@class_autologger
class TestDataset:
    logger: logging.Logger

    @silent
    def test_len_accuracy(self, mock_dataset_data: MtxList) -> None:
        ds = Dataset(mock_dataset_data)
        current_len = len(ds)

        if current_len != len(mock_dataset_data):
            self.logger.error(
                f"[test_len_accuracy] Length mismatch. Expected {len(mock_dataset_data)}, got {current_len}."
            )
        assert current_len == len(mock_dataset_data)

    def test_getitem_valid_index(self, mock_dataset_data: MtxList) -> None:
        ds = Dataset(mock_dataset_data)
        idx = 2
        item = ds[idx]

        if item is not None:
            self.logger.debug(
                f"[test_getitem_valid_index] Successfully retrieved item at index {idx}."
            )
        assert np.array_equal(item, mock_dataset_data[idx])

    def test_getitem_out_of_bounds(self, mock_dataset_data: MtxList) -> None:
        ds = Dataset(mock_dataset_data)
        bad_idx = 10

        self.logger.info(
            f"[test_getitem_out_of_bounds] Testing boundary safety for index {bad_idx}."
        )
        with pytest.raises(IndexError):
            _ = ds[bad_idx]

    @silent
    def test_getitem_negative_index(self, mock_dataset_data: MtxList) -> None:
        ds = Dataset(mock_dataset_data)
        idx = -1

        try:
            item = ds[idx]
            self.logger.debug(
                f"[test_getitem_negative_index] Negative indexing returned data for index {idx}."
            )
            assert np.array_equal(item, mock_dataset_data[idx])
        except IndexError:
            self.logger.warning(
                f"[test_getitem_negative_index] Dataset does not support negative indexing: {idx}."
            )
            raise
