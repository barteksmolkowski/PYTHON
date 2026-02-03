import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from preprocessing import ConvolutionActions

logger = logging.getLogger(__name__)


class TestConvolutionActions:
    @pytest.fixture
    def mock_data(self):
        channel = np.ones((10, 10), dtype=np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        return channel, kernel

    def test_convolution_2d(self, mock_data):
        logger.info("ACTION: Testing convolution_2d")
        ca = ConvolutionActions()
        channel, kernel = mock_data

        results = ca.convolution_2d(channel, filters=[kernel])

        if not (isinstance(results, list) and len(results) == 1):
            logger.error("Assertion failed: convolution_2d result is not a list of 1")
        assert isinstance(results, list) and len(results) == 1

        res = results[0]
        if res.shape != (8, 8):
            logger.error(
                f"Assertion failed: convolution_2d output shape mismatch: {res.shape}"
            )
        assert res.shape == (8, 8)

        if res.dtype != np.uint8:
            logger.error(
                f"Assertion failed: convolution_2d dtype is {res.dtype}, expected uint8"
            )
        assert res.dtype == np.uint8

        expected_sum = 8 * 8 * 9
        if np.sum(res) != expected_sum:
            logger.error(f"Assertion failed: convolution_2d sum mismatch")
        assert np.sum(res) == expected_sum

        logger.info("SUCCESS: convolution_2d verified")

    def test_apply_filters(self, mock_data):
        logger.info("ACTION: Testing apply_filters")
        ca = ConvolutionActions()

        ca.geometry = MagicMock()
        channel, kernel = mock_data
        ca.geometry.pad.return_value = np.ones((12, 12), dtype=np.uint8)

        results = ca.apply_filters(channels=[channel], filters=[kernel])

        if not (isinstance(results, list) and len(results) == 1):
            logger.error("Assertion failed: apply_filters result length mismatch")
        assert isinstance(results, list) and len(results) == 1

        res = results[0]
        if res.shape != (10, 10):
            logger.error(
                f"Assertion failed: apply_filters output shape mismatch: {res.shape}"
            )
        assert res.shape == (10, 10)

        if res.dtype != np.uint8:
            logger.error("Assertion failed: apply_filters dtype is not uint8")
        assert res.dtype == np.uint8

        logger.info("SUCCESS: apply_filters verified")
