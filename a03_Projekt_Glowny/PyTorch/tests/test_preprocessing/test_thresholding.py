import logging
from unittest.mock import patch

import numpy as np
import pytest
from preprocessing import Thresholding

logger = logging.getLogger(__name__)


class TestThresholding:
    @pytest.fixture
    def mock_mtx(self):
        matrix = np.full((10, 10), 200, dtype=np.uint8)
        matrix[4:6, 4:6] = 50
        return matrix

    def test_adaptive_threshold(self, mock_mtx):
        logger.info("ACTION: Testing adaptive_threshold")

        thresholding = Thresholding()
        mock_kernel = np.ones((5, 5), dtype=np.float32)

        with patch.object(
            Thresholding, "_generate_gaussian_kernel", return_value=mock_kernel
        ):
            result = thresholding.adaptive_threshold(
                mock_mtx, block_size=5, c=2, auto_params=False
            )

            if result.shape != (10, 10):
                logger.error(
                    f"Assertion failed: result shape {result.shape} != (10, 10)"
                )
            assert result.shape == (10, 10)

            if result.dtype != np.uint8:
                logger.error(f"Assertion failed: result dtype {result.dtype} != uint8")
            assert result.dtype == np.uint8

            unique_values = np.unique(result)
            if not np.all(np.isin(unique_values, [0, 255])):
                logger.error(
                    f"Assertion failed: result contains values other than 0 and 255: {unique_values}"
                )
            assert np.all(np.isin(unique_values, [0, 255]))

            total_sum = np.sum(result)
            if total_sum >= 100 * 255:
                logger.error(
                    "Assertion failed: thresholding did not create any black pixels"
                )
            assert total_sum < 100 * 255

            logger.info("SUCCESS: adaptive_threshold verified")

    def test_adaptive_threshold_auto_params(self, mock_mtx):
        logger.info("ACTION: Testing adaptive_threshold with auto_params")

        thresholding = Thresholding()

        result = thresholding.adaptive_threshold(mock_mtx, auto_params=True)

        if result.shape != (10, 10):
            logger.error("Assertion failed: auto_params result shape mismatch")
        assert result.shape == (10, 10)

        if result.dtype != np.uint8:
            logger.error("Assertion failed: auto_params result dtype is not uint8")
        assert result.dtype == np.uint8

        logger.info("SUCCESS: adaptive_threshold with auto_params verified")
