import logging

import numpy as np
import pytest

from preprocessing import GrayScaleProcessing

logger = logging.getLogger(__name__)


class TestGrayScaleProcessing:
    @pytest.fixture
    def mock_rgb(self):
        return np.ones((10, 10, 3), dtype=np.uint8)

    @pytest.fixture
    def mock_gray(self):
        return np.ones((10, 10), dtype=np.uint8)

    @pytest.fixture
    def mock_precision_rgb(self):
        return np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)

    @pytest.fixture
    def mock_rgba(self):
        return np.ones((10, 10, 4), dtype=np.uint8)

    @pytest.fixture
    def mock_empty(self):
        return np.array([[]], dtype=np.uint8)

    def test_convert_color_space_to_gray(self, mock_rgb):
        logger.info("ACTION: Testing convert_color_space [to_gray=True]")
        processor = GrayScaleProcessing()
        result = processor.convert_color_space(mock_rgb, to_gray=True)

        if result.shape != (10, 10):
            logger.error(f"Assertion failed: result shape {result.shape} is not 2D")
        assert result.shape == (10, 10)

        if result.dtype != np.uint8:
            logger.error(f"Assertion failed: dtype {result.dtype} is not uint8")
        assert result.dtype == np.uint8

        if np.sum(result) != 100:
            logger.error(f"Assertion failed: sum {np.sum(result)} != 100")
        assert np.sum(result) == 100
        logger.info("SUCCESS: convert_color_space [to_gray=True] verified")

    def test_convert_color_space_to_color(self, mock_gray):
        logger.info("ACTION: Testing convert_color_space [to_gray=False]")
        processor = GrayScaleProcessing()
        result = processor.convert_color_space(mock_gray, to_gray=False)

        if result.shape != (10, 10, 3):
            logger.error(f"Assertion failed: result shape {result.shape} is not 3D")
        assert result.shape == (10, 10, 3)

        if result.dtype != np.uint8:
            logger.error(f"Assertion failed: dtype {result.dtype} is not uint8")
        assert result.dtype == np.uint8

        if np.sum(result) != 300:
            logger.error(f"Assertion failed: sum {np.sum(result)} != 300")
        assert np.sum(result) == 300
        logger.info("SUCCESS: convert_color_space [to_gray=False] verified")

    def test_convert_color_space_precision(self, mock_precision_rgb):
        logger.info("ACTION: Testing convert_color_space precision")
        processor = GrayScaleProcessing()
        result = processor.convert_color_space(mock_precision_rgb, to_gray=True)

        expected = np.array([[76, 149]], dtype=np.uint8)
        if not np.array_equal(result, expected):
            logger.error(f"Assertion failed: Precision mismatch. Got {result}")
        assert np.array_equal(result, expected)
        logger.info("SUCCESS: convert_color_space precision verified")

    def test_convert_color_space_float_input(self):
        logger.info("ACTION: Testing convert_color_space float input")
        processor = GrayScaleProcessing()
        float_mtx = np.array([[[100.7, 100.7, 100.7]]], dtype=np.float32)
        result = processor.convert_color_space(float_mtx, to_gray=True)

        if result[0, 0] != 100 or result.dtype != np.uint8:
            logger.error(f"Assertion failed: Float cast failed, got {result[0, 0]}")
        assert result[0, 0] == 100 and result.dtype == np.uint8
        logger.info("SUCCESS: convert_color_space float input verified")

    def test_convert_color_space_rgba(self, mock_rgba):
        logger.info("ACTION: Testing convert_color_space RGBA")
        processor = GrayScaleProcessing()
        result = processor.convert_color_space(mock_rgba, to_gray=True)

        if result.shape != (10, 10):
            logger.error(f"Assertion failed: RGBA ignore failed, shape: {result.shape}")
        assert result.shape == (10, 10)
        logger.info("SUCCESS: convert_color_space RGBA verified")

    def test_convert_color_space_empty(self, mock_empty):
        logger.info("ACTION: Testing convert_color_space empty")
        processor = GrayScaleProcessing()
        result = processor.convert_color_space(mock_empty, to_gray=True)

        if result.size != 0:
            logger.error("Assertion failed: result not empty")
        assert result.size == 0
        logger.info("SUCCESS: convert_color_space empty verified")
