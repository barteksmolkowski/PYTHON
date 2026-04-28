import logging
from typing import TypeAlias

import numpy as np
from common_utils import class_autologger, silent

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


@class_autologger
class TestBatchProcessing:
    logger: logging.Logger

    def test_create_batches_logic(self, batch_proc, mock_mtx_list):
        batches = batch_proc.create_batches(mock_mtx_list, batch_size=2, shuffle=False)

        if len(batches) != 3:
            self.logger.error(
                f"[test_create_batches_logic] Logic error! Got {len(batches)} batches."
            )

        last_batch_len = len(batches[-1])
        assert len(batches) == 3
        assert last_batch_len == 1
        self.logger.info(
            f"[test_create_batches_logic] Validated {len(mock_mtx_list)} items."
        )

    def test_process_batch_validation(self, batch_proc, test_paths):
        results = batch_proc.process_batch(test_paths)
        assert isinstance(results, list)
        self.logger.debug(
            f"[test_process_batch_validation] Checked {len(test_paths)} paths."
        )

    @silent
    def test_shuffle_reproducibility(self, batch_proc, mock_mtx_list):
        batches = batch_proc.create_batches(mock_mtx_list, batch_size=1, shuffle=True)
        total = sum(len(b) for b in batches)

        if total != len(mock_mtx_list):
            self.logger.error(
                f"[test_shuffle_reproducibility] Data loss: {total} vs {len(mock_mtx_list)}"
            )

        assert total == len(mock_mtx_list)
