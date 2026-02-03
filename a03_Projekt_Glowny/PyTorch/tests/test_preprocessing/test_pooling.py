import logging

import numpy as np
import pytest

from preprocessing import Pooling

logger = logging.getLogger(__name__)

class TestPooling:
    @pytest.fixture
    def mock_mtx(self):
        return np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.uint8)

    def test_max_pool(self, mock_mtx):
        logger.info("ACTION: Testing max_pool")
        
        pooling = Pooling()
        kernel_size = (2, 2)
        stride = 2

        result = pooling.max_pool(mock_mtx, kernel_size=kernel_size, stride=stride)

        if result.shape != (2, 2):
            logger.error(f"Assertion failed: max_pool output shape {result.shape} != (2, 2)")
        assert result.shape == (2, 2)

        if result.dtype != np.uint8:
            logger.error(f"Assertion failed: max_pool dtype {result.dtype} != uint8")
        assert result.dtype == np.uint8

        expected = np.array([[6, 8], [14, 16]], dtype=np.uint8)
        if not np.array_equal(result, expected):
            logger.error(f"Assertion failed: max_pool values mismatch. Got {result}")
        assert np.array_equal(result, expected)

        if np.sum(result) != (6 + 8 + 14 + 16):
            logger.error(f"Assertion failed: max_pool sum {np.sum(result)} mismatch")
        assert np.sum(result) == 44

        logger.info("SUCCESS: max_pool verified")

    def test_max_pool_with_padding(self, mock_mtx):
        logger.info("ACTION: Testing max_pool with padding")
        
        pooling = Pooling()
        result = pooling.max_pool(mock_mtx, kernel_size=(2, 2), stride=2, pad_width=1, pad_values=0)

        if result.shape != (3, 3):
            logger.error(f"Assertion failed: padded max_pool shape {result.shape} != (3, 3)")
        assert result.shape == (3, 3)
        
        if result.dtype != np.uint8:
            logger.error("Assertion failed: padded result dtype is not uint8")
        assert result.dtype == np.uint8

        logger.info("SUCCESS: max_pool with padding verified")
