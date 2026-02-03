import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from preprocessing import ImageHandler

logger = logging.getLogger(__name__)

class TestImageHandler:
    @pytest.fixture
    def mock_img_data(self):
        return np.ones((10, 20, 3), dtype=np.uint8)

    def test_open_image(self, mock_img_data):
        logger.info("ACTION: Testing open_image")
        handler = ImageHandler()
        
        mock_pil = MagicMock(spec=Image.Image)
        mock_pil.size = (20, 10)
        mock_pil.convert.return_value = mock_pil
        mock_pil.__enter__.return_value = mock_pil
        
        with patch("PIL.Image.open", return_value=mock_pil), \
             patch("numpy.array", return_value=mock_img_data):
            
            array, w, h = handler.open_image("test.jpg")

            if array.shape != (10, 20, 3) or array.dtype != np.uint8:
                logger.error(f"Assertion failed: array properties mismatch {array.shape}")
            assert array.shape == (10, 20, 3)
            assert array.dtype == np.uint8

            if w != 20 or h != 10:
                logger.error(f"Assertion failed: dimensions mismatch w:{w} h:{h}")
            assert w == 20 and h == 10

        logger.info("SUCCESS: open_image verified")

    def test_save(self, mock_img_data):
        logger.info("ACTION: Testing save")
        handler = ImageHandler()
        
        with patch("PIL.Image.fromarray") as mock_fromarray:
            mock_pil_instance = MagicMock()
            mock_fromarray.return_value = mock_pil_instance
            
            handler.save(mock_img_data, "save.png")

            if not mock_fromarray.called:
                logger.error("Assertion failed: Image.fromarray was not called")
            assert mock_fromarray.called

            if not mock_pil_instance.save.called:
                logger.error("Assertion failed: PIL save method was not called")
            assert mock_pil_instance.save.called
            
        logger.info("SUCCESS: save verified")

    def test_handle_file_read(self, mock_img_data):
        logger.info("ACTION: Testing handle_file [read mode]")
        handler = ImageHandler()
        
        with patch.object(ImageHandler, "open_image", return_value=(mock_img_data, 20, 10)):
            res = handler.handle_file("test.png", is_save_mode=False)

            if res is None or res.shape != (10, 20, 3):
                logger.error("Assertion failed: handle_file read returned invalid data")
            assert res.shape == (10, 20, 3)
            assert res.dtype == np.uint8

        logger.info("SUCCESS: handle_file [read mode] verified")

    def test_handle_file_save(self, mock_img_data):
        logger.info("ACTION: Testing handle_file [save mode]")
        handler = ImageHandler()
        
        with patch.object(ImageHandler, "save") as mock_save:
            handler.handle_file("test.png", data=mock_img_data, is_save_mode=True)

            if not mock_save.called:
                logger.error("Assertion failed: save method was not called in save mode")
            assert mock_save.called

        logger.info("SUCCESS: handle_file [save mode] verified")

    def test_handle_file_error(self):
        logger.info("ACTION: Testing handle_file [Value Error]")
        handler = ImageHandler()
        
        try:
            with pytest.raises(ValueError):
                handler.handle_file("test.png", data=None, is_save_mode=True)
        except Exception as e:
            logger.error(f"Assertion failed: ValueError not raised as expected: {e}")
            raise e
            
        logger.info("SUCCESS: handle_file [Value Error] verified")