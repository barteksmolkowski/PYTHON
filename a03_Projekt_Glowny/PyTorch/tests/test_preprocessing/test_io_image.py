import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from preprocessing import ImageHandler

logger = logging.getLogger("test_logger")


class TestImageHandler:
    def test_handle_file_error(self, image_handler):
        logger.info(
            "[test_handle_file_error] ACTION: Testing handle_file [Value Error]"
        )

        if True:  # Logic wrapper for error logging
            with pytest.raises(ValueError):
                image_handler.handle_file("test.png", data=None, is_save_mode=True)

        logger.info("[test_handle_file_error] Validated 1 items.")

    def test_handle_file_save(self, image_handler, mock_img_data):
        logger.info("[test_handle_file_save] ACTION: Testing handle_file [save mode]")

        with patch.object(ImageHandler, "save") as mock_save:
            image_handler.handle_file("test.png", data=mock_img_data, is_save_mode=True)

            if not mock_save.called:
                logger.error(
                    "[test_handle_file_save] Validation error: save method was not called"
                )
            assert mock_save.called

        logger.info("[test_handle_file_save] Validated 1 items.")

    def test_handle_file_read(self, image_handler, mock_img_data):
        logger.info("[test_handle_file_read] ACTION: Testing handle_file [read mode]")

        with patch.object(
            ImageHandler, "open_image", return_value=(mock_img_data, 20, 10)
        ):
            res = image_handler.handle_file("test.png", is_save_mode=False)

            if res is None or res.shape != (10, 20, 3):
                logger.error(
                    f"[test_handle_file_read] Data loss: handle_file read returned invalid shape {getattr(res, 'shape', None)}"
                )

            assert res.shape == (10, 20, 3)
            assert res.dtype == np.uint8

        logger.info("[test_handle_file_read] Validated 1 items.")

    def test_save(self, image_handler, mock_img_data):
        logger.info("[test_save] ACTION: Testing save")

        with patch("PIL.Image.fromarray") as mock_fromarray:
            mock_pil_instance = MagicMock()
            mock_fromarray.return_value = mock_pil_instance

            image_handler.save(mock_img_data, "save.png")

            if not mock_fromarray.called:
                logger.error(
                    "[test_save] Validation error: Image.fromarray was not called"
                )
            assert mock_fromarray.called

            if not mock_pil_instance.save.called:
                logger.error(
                    "[test_save] Validation error: PIL save method was not called"
                )
            assert mock_pil_instance.save.called

        logger.info("[test_save] Validated 1 items.")

    def test_open_image(self, image_handler, mock_img_data):
        logger.info("[test_open_image] ACTION: Testing open_image")

        mock_pil = MagicMock(spec=Image.Image)
        mock_pil.size = (20, 10)
        mock_pil.convert.return_value = mock_pil
        mock_pil.__enter__.return_value = mock_pil

        with (
            patch("PIL.Image.open", return_value=mock_pil),
            patch("numpy.array", return_value=mock_img_data),
        ):
            array, w, h = image_handler.open_image("test.jpg")

            if array.shape != (10, 20, 3) or array.dtype != np.uint8:
                logger.error(
                    f"[test_open_image] Physical mismatch: shape={array.shape}, dtype={array.dtype}"
                )

            assert array.shape == (10, 20, 3)
            assert array.dtype == np.uint8

            if w != 20 or h != 10:
                logger.error(
                    f"[test_open_image] Out of bounds: dimensions mismatch w:{w} h:{h}"
                )

            assert w == 20 and h == 10

        logger.info("[test_open_image] Validated 1 items.")
