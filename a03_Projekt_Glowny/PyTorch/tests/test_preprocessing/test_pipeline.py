import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from preprocessing import ImageDataPreprocessing, TransformPipeline

logger = logging.getLogger(__name__)

class TestTransformPipeline:
    @pytest.fixture
    def mock_sample(self):
        return np.ones((28, 28), dtype=np.uint8)

    def test_apply(self, mock_sample):
        logger.info("ACTION: Testing TransformPipeline.apply")
        
        tp = TransformPipeline()
        tp.grayscale = MagicMock()
        tp.geometry = MagicMock()
        tp.augmentation = MagicMock()
        tp.normalization = MagicMock()

        tp.grayscale.convert_color_space.return_value = mock_sample
        tp.geometry.prepare_standard_geometry.return_value = mock_sample
        tp.augmentation.augment.return_value = [mock_sample, mock_sample]
        tp.normalization.process.return_value = mock_sample.astype(np.float32)

        result = tp.apply(mock_sample)

        if not isinstance(result, list) or len(result) != 2:
            logger.error("Assertion failed: apply result should be a list of 2 augmented samples")
        assert isinstance(result, list) and len(result) == 2

        for i, sample in enumerate(result):
            if sample.shape != (28, 28):
                logger.error(f"Assertion failed: sample {i} shape mismatch")
            assert sample.shape == (28, 28)
            
            if sample.dtype != np.float32:
                logger.warning(f"Note: sample {i} is float32 due to normalization")

        logger.info("SUCCESS: TransformPipeline.apply verified")


class TestImageDataPreprocessing:
    @pytest.fixture
    def mock_batch(self):
        return [np.ones((28, 28), dtype=np.uint8)]

    def test_preprocess(self, mock_batch):
        logger.info("ACTION: Testing ImageDataPreprocessing.preprocess")
        
        idp = ImageDataPreprocessing()
        idp.converter = MagicMock()
        idp.pipeline = MagicMock()
        
        idp.converter.get_channels_from_file.return_value = [np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10))]
        idp.pipeline.apply.return_value = mock_batch

        result = idp.preprocess("fake_path.png")

        if result is None:
            logger.error("Assertion failed: preprocess returned None")
        assert result is not None

        if len(result) != 3:
            logger.error(f"Assertion failed: expected 3 channel results, got {len(result)}")
        assert len(result) == 3

        for i, channel_batch in enumerate(result):
            if not isinstance(channel_batch, list):
                logger.error(f"Assertion failed: result for channel {i} is not a list")
            assert isinstance(channel_batch, list)

        logger.info("SUCCESS: ImageDataPreprocessing.preprocess verified")

    def test_preprocess_exception(self):
        logger.info("ACTION: Testing ImageDataPreprocessing.preprocess [Exception handling]")
        
        idp = ImageDataPreprocessing()
        idp.converter = MagicMock()
        idp.converter.get_channels_from_file.side_effect = Exception("Mock Error")

        result = idp.preprocess("invalid_path.png")

        if result is not None:
            logger.error("Assertion failed: preprocess should return None on exception")
        assert result is None
        
        logger.info("SUCCESS: ImageDataPreprocessing.preprocess [Exception handling] verified")
