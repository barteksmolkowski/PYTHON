import logging
from unittest.mock import patch

import numpy as np
import pytest

from preprocessing import ImageGeometry

logger = logging.getLogger(__name__)

class TestImageGeometry:
    @pytest.fixture
    def mock_mtx(self):
        return np.ones((50, 50), dtype=np.uint8)

    def test_resize(self, mock_mtx):
        logger.info("ACTION: Testing resize")
        
        geo = ImageGeometry()
        target_size = (28, 28)
        
        with patch.object(ImageGeometry, '_downscale_vectorized', return_value=np.ones(target_size, dtype=np.uint8)) as mock_down:
            res = geo.resize(mock_mtx, new_size=target_size)

            if res.shape != target_size:
                logger.error(f"Assertion failed: resize shape {res.shape} != {target_size}")
            assert res.shape == target_size

            if res.dtype != np.uint8:
                logger.error(f"Assertion failed: resize dtype {res.dtype} is not uint8")
            assert res.dtype == np.uint8

            if np.sum(res) != (28 * 28):
                logger.error("Assertion failed: resize sum mismatch")
            assert np.sum(res) == (28 * 28)

        logger.info("SUCCESS: resize verified")

    def test_prepare_standard_geometry(self, mock_mtx):
        logger.info("ACTION: Testing prepare_standard_geometry")
        
        geo = ImageGeometry()
        target_size = (28, 28)
        padding = 2
        inner_size = (24, 24)
        
        with patch.object(ImageGeometry, 'resize', return_value=np.ones(inner_size, dtype=np.uint8)):
            res = geo.prepare_standard_geometry(mock_mtx, target_size=target_size, padding=padding, pad_value=0)

            if res.shape != target_size:
                logger.error(f"Assertion failed: standard_geometry shape {res.shape} != {target_size}")
            assert res.shape == target_size

            if res.dtype != np.uint8:
                logger.error(f"Assertion failed: standard_geometry dtype {res.dtype} is not uint8")
            assert res.dtype == np.uint8

            expected_sum = 24 * 24 
            if np.sum(res) != expected_sum:
                logger.error(f"Assertion failed: standard_geometry sum {np.sum(res)} != {expected_sum}")
            assert np.sum(res) == expected_sum

        logger.info("SUCCESS: prepare_standard_geometry verified")
