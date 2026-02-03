import logging

import numpy as np
import pytest

from preprocessing import Normalization

logger = logging.getLogger(__name__)

class TestNormalization:
    @pytest.fixture
    def mock_mtx(self):
        return np.array([[10, 20], [30, 40]], dtype=np.uint8)

    def test_normalize(self, mock_mtx):
        logger.info("ACTION: Testing normalize")
        norm = Normalization()
        
        res = norm.normalize(mock_mtx, old_r=(0, 100), new_r=(0, 1))

        if res.shape != (2, 2):
            logger.error(f"Assertion failed: normalize shape mismatch {res.shape}")
        assert res.shape == (2, 2)

        if not np.isclose(np.sum(res), 1.0):
            logger.error(f"Assertion failed: normalize sum {np.sum(res)} != 1.0")
        assert np.isclose(np.sum(res), 1.0)
        
        logger.info("SUCCESS: normalize verified")

    def test_z_score_normalization(self, mock_mtx):
        logger.info("ACTION: Testing z_score_normalization")
        norm = Normalization()
        
        res = norm.z_score_normalization(mock_mtx)

        if res.shape != (2, 2):
            logger.error(f"Assertion failed: z_score shape mismatch")
        assert res.shape == (2, 2)

        if not np.isclose(np.mean(res), 0.0):
            logger.error(f"Assertion failed: z_score mean {np.mean(res)} != 0")
        assert np.isclose(np.mean(res), 0.0, atol=1e-7)

        if not np.isclose(np.std(res), 1.0):
            logger.error(f"Assertion failed: z_score std {np.std(res)} != 1")
        assert np.isclose(np.std(res), 1.0, atol=1e-7)
        
        logger.info("SUCCESS: z_score_normalization verified")

    def test_process(self, mock_mtx):
        logger.info("ACTION: Testing process")
        norm = Normalization()
        
        res_minmax = norm.process(mock_mtx, use_z_score=False, old_r=(0, 40), new_r=(0, 1))

        if res_minmax.shape != (2, 2):
            logger.error("Assertion failed: process shape mismatch")
        assert res_minmax.shape == (2, 2)

        if res_minmax[1, 1] != 1.0:
            logger.error(f"Assertion failed: process minmax max value mismatch: {res_minmax[1, 1]}")
        assert res_minmax[1, 1] == 1.0

        logger.info("SUCCESS: process verified")

    def test_normalize_zero_denominator(self):
        logger.info("ACTION: Testing normalize zero denominator edge case")
        norm = Normalization()
        mtx = np.ones((2, 2))
        
        res = norm.normalize(mtx, old_r=(10, 10), new_r=(0, 1))
        
        if not np.all(res == 0):
            logger.error("Assertion failed: zero denominator did not return new_r[0]")
        assert np.all(res == 0)
        
        logger.info("SUCCESS: normalize zero denominator verified")
