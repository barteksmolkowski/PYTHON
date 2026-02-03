import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from preprocessing import ImageToMatrixConverter

logger = logging.getLogger(__name__)


class TestImageToMatrixConverter:
    @pytest.fixture
    def mock_mtx(self):
        return np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)

    @pytest.fixture
    def mock_io(self, mock_mtx):
        mock_img = MagicMock(spec=Image.Image)
        mock_img.convert.return_value = mock_img
        mock_img.__enter__.return_value = mock_img
        with patch("PIL.Image.open", return_value=mock_img), patch(
            "numpy.array", return_value=mock_mtx
        ):
            yield mock_mtx

    def test_get_channels_from_file(self, mock_io):
        logger.info("ACTION: Testing get_channels_from_file")

        converter = ImageToMatrixConverter()
        expected_mtx = mock_io

        result = converter.get_channels_from_file("test.png")

        if not (isinstance(result, list) and len(result) == 3):
            logger.error(
                "Assertion failed: result must be a list of 3 channel matrices"
            )
        assert isinstance(result, list) and len(result) == 3

        for i, channel in enumerate(result):
            if channel.shape != (16, 16):
                logger.error(
                    f"Assertion failed: channel {i} has incorrect shape {channel.shape}"
                )
            assert channel.shape == (16, 16)

            if channel.dtype != np.uint8:
                logger.error(f"Assertion failed: channel {i} dtype is not uint8")
            assert channel.dtype == np.uint8

            expected_sum = np.sum(expected_mtx[..., i])
            if np.sum(channel) != expected_sum:
                logger.error(f"Assertion failed: channel {i} sum mismatch")
            assert np.sum(channel) == expected_sum

        logger.info("SUCCESS: get_channels_from_file verified")
