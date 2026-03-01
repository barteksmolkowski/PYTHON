import logging

import numpy as np

logger = logging.getLogger(__name__)


class TestImageToMatrixConverter:
    def test_get_channels_from_file(self, converter_engine, mock_io):
        logger.info(
            "[test_get_channels_from_file] ACTION: Testing get_channels_from_file"
        )

        expected_mtx = mock_io
        result = converter_engine.get_channels_from_file("test.png")

        if not (isinstance(result, list) and len(result) == 3):
            logger.error(
                f"[test_get_channels_from_file] Data loss: expected 3 channels, got {type(result)}"
            )

        assert isinstance(result, list)
        assert len(result) == 3

        for i, channel in enumerate(result):
            if channel.shape != (16, 16):
                logger.error(
                    f"[test_get_channels_from_file] Out of bounds: channel {i} shape {channel.shape}"
                )
            assert channel.shape == (16, 16)

            if channel.dtype != np.uint8:
                logger.error(
                    f"[test_get_channels_from_file] Logic error: channel {i} dtype is {channel.dtype}"
                )
            assert channel.dtype == np.uint8

            expected_sum = np.sum(expected_mtx[..., i])
            actual_sum = np.sum(channel)
            if actual_sum != expected_sum:
                logger.error(
                    f"[test_get_channels_from_file] Data loss: channel {i} sum mismatch. {actual_sum} vs {expected_sum}"
                )
            assert actual_sum == expected_sum

        logger.info(f"[test_get_channels_from_file] Validated {len(result)} items.")
