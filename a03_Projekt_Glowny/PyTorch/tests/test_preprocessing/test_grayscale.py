import logging

import numpy as np

logger = logging.getLogger("test_logger")


class TestGrayScaleProcessing:
    def test_convert_color_space_rgba(self, grayscale_engine, mock_rgba):
        logger.info(
            "[test_convert_color_space_rgba] ACTION: Testing convert_color_space RGBA"
        )
        result = grayscale_engine.convert_color_space(mock_rgba, to_gray=True)

        if result.shape != (10, 10):
            logger.error(
                f"[test_convert_color_space_rgba] Out of bounds: RGBA ignore failed, shape={result.shape}"
            )
        assert result.shape == (10, 10)
        logger.info("[test_convert_color_space_rgba] Validated 1 items.")

    def test_convert_color_space_precision(self, grayscale_engine, mock_precision_rgb):
        logger.info(
            "[test_convert_color_space_precision] ACTION: Testing convert_color_space precision"
        )
        result = grayscale_engine.convert_color_space(mock_precision_rgb, to_gray=True)

        expected = np.array([[76, 149]], dtype=np.uint8)
        if not np.array_equal(result, expected):
            logger.error(
                f"[test_convert_color_space_precision] Data loss: Precision mismatch. Got {result}"
            )
        assert np.array_equal(result, expected)
        logger.info("[test_convert_color_space_precision] Validated 1 items.")

    def test_convert_color_space_float_input(self, grayscale_engine):
        logger.info(
            "[test_convert_color_space_float_input] ACTION: Testing convert_color_space float input"
        )
        float_mtx = np.array([[[100.7, 100.7, 100.7]]], dtype=np.float32)
        result = grayscale_engine.convert_color_space(float_mtx, to_gray=True)

        if result[0, 0] != 100 or result.dtype != np.uint8:
            logger.error(
                f"[test_convert_color_space_float_input] Logic error: Float cast failed, got {result[0, 0]}"
            )
        assert result[0, 0] == 100 and result.dtype == np.uint8
        logger.info("[test_convert_color_space_float_input] Validated 1 items.")

    def test_convert_color_space_empty(self, grayscale_engine, mock_empty_mtx):
        logger.info(
            "[test_convert_color_space_empty] ACTION: Testing convert_color_space empty"
        )
        result = grayscale_engine.convert_color_space(mock_empty_mtx, to_gray=True)

        if result.size != 0:
            logger.error(
                f"[test_convert_color_space_empty] Validation error: result not empty, size={result.size}"
            )
        assert result.size == 0
        logger.info("[test_convert_color_space_empty] Validated 1 items.")

    def test_convert_color_space_to_gray(self, grayscale_engine, mock_rgb):
        logger.info(
            "[test_convert_color_space_to_gray] ACTION: Testing convert_color_space [to_gray=True]"
        )
        result = grayscale_engine.convert_color_space(mock_rgb, to_gray=True)

        if result.shape != (10, 10) or result.dtype != np.uint8:
            logger.error(
                f"[test_convert_color_space_to_gray] Out of bounds: shape={result.shape}, dtype={result.dtype}"
            )
        assert result.shape == (10, 10)
        assert result.dtype == np.uint8

        actual_sum = np.sum(result)
        if actual_sum != 100:
            logger.error(
                f"[test_convert_color_space_to_gray] Data loss: sum {actual_sum} != 100"
            )
        assert actual_sum == 100
        logger.info("[test_convert_color_space_to_gray] Validated 1 items.")

    def test_convert_color_space_to_color(self, grayscale_engine, mock_gray):
        logger.info(
            "[test_convert_color_space_to_color] ACTION: Testing convert_color_space [to_gray=False]"
        )
        result = grayscale_engine.convert_color_space(mock_gray, to_gray=False)

        if result.shape != (10, 10, 3) or result.dtype != np.uint8:
            logger.error(
                f"[test_convert_color_space_to_color] Out of bounds: shape={result.shape}, dtype={result.dtype}"
            )
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8

        actual_sum = np.sum(result)
        if actual_sum != 300:
            logger.error(
                f"[test_convert_color_space_to_color] Data loss: sum {actual_sum} != 300"
            )
        assert actual_sum == 300
        logger.info("[test_convert_color_space_to_color] Validated 1 items.")
