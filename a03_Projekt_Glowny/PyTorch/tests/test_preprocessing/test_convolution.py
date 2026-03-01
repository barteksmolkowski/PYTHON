import logging

import numpy as np

logger = logging.getLogger(__name__)


class TestConvolutionActions:
    def test_apply_filters(self, conv_engine, mock_conv_data):
        logger.info("[test_apply_filters] ACTION: Testing apply_filters")
        channel, kernel = mock_conv_data
        conv_engine.geometry.pad.return_value = np.ones((12, 12), dtype=np.uint8)

        results = conv_engine.apply_filters(channels=[channel], filters=[kernel])

        if not (isinstance(results, list) and len(results) == 1):
            logger.error(
                f"[test_apply_filters] Data loss: result length mismatch. Expected 1, got {len(results) if isinstance(results, list) else type(results)}"
            )
        assert isinstance(results, list) and len(results) == 1

        res = results[0]
        if res.shape != (10, 10) or res.dtype != np.uint8:
            logger.error(
                f"[test_apply_filters] Validation error: shape={res.shape}, dtype={res.dtype}"
            )

        assert res.shape == (10, 10)
        assert res.dtype == np.uint8
        logger.info("[test_apply_filters] Validated 1 items.")

    def test_convolution_2d(self, conv_engine, mock_conv_data):
        logger.info("[test_convolution_2d] ACTION: Testing convolution_2d")
        channel, kernel = mock_conv_data

        results = conv_engine.convolution_2d(channel, filters=[kernel])

        if not (isinstance(results, list) and len(results) == 1):
            logger.error(
                f"[test_convolution_2d] Data loss: result is not a list of 1. Got {type(results)}"
            )
        assert isinstance(results, list) and len(results) == 1

        res = results[0]
        if res.shape != (8, 8):
            logger.error(
                f"[test_convolution_2d] Out of bounds: expected (8, 8), got {res.shape}"
            )
        assert res.shape == (8, 8)

        if res.dtype != np.uint8:
            logger.error(
                f"[test_convolution_2d] Logic error: dtype is {res.dtype}, expected uint8"
            )
        assert res.dtype == np.uint8

        expected_sum = 8 * 8 * 9
        actual_sum = np.sum(res)
        if actual_sum != expected_sum:
            logger.error(
                f"[test_convolution_2d] Data loss: sum mismatch. Expected {expected_sum}, got {actual_sum}"
            )
        assert actual_sum == expected_sum
        logger.info("[test_convolution_2d] Validated 1 items.")
