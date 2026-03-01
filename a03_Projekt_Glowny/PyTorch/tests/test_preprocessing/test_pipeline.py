import logging

import numpy as np

logger = logging.getLogger("test_logger")


class TestTransformPipeline:
    def test_apply(self, pipeline_engine, mock_28x28_sample):
        logger.info("[test_apply] ACTION: Testing TransformPipeline.apply")

        pipeline_engine.grayscale.convert_color_space.return_value = mock_28x28_sample
        pipeline_engine.geometry.prepare_standard_geometry.return_value = (
            mock_28x28_sample
        )
        pipeline_engine.augmentation.augment.return_value = [
            mock_28x28_sample,
            mock_28x28_sample,
        ]
        pipeline_engine.normalization.process.return_value = mock_28x28_sample.astype(
            np.float32
        )

        result = pipeline_engine.apply(mock_28x28_sample)

        if not isinstance(result, list) or len(result) != 2:
            logger.error(
                f"[test_apply] Data loss: expected list of 2, got {type(result)} size {len(result) if isinstance(result, list) else 'N/A'}"
            )

        assert isinstance(result, list)
        assert len(result) == 2

        for i, sample in enumerate(result):
            if sample.shape != (28, 28):
                logger.error(
                    f"[test_apply] Out of bounds: sample {i} shape mismatch: {sample.shape}"
                )
            assert sample.shape == (28, 28)

            if sample.dtype != np.float32:
                logger.warning(
                    f"[test_apply] Unexpected dtype: sample {i} is {sample.dtype}, expected float32"
                )
            assert sample.dtype == np.float32

        logger.info(f"[test_apply] Validated {len(result)} items.")


class TestImageDataPreprocessing:
    def test_preprocess_exception(self, idp_engine):
        logger.info(
            "[test_preprocess_exception] ACTION: Testing ImageDataPreprocessing.preprocess [Exception]"
        )

        idp_engine.converter.get_channels_from_file.side_effect = Exception(
            "Mock Error"
        )

        result = idp_engine.preprocess("invalid_path.png")

        if result is not None:
            logger.error(
                "[test_preprocess_exception] Logic error: preprocess should return None on failure"
            )

        assert result is None
        logger.info("[test_preprocess_exception] Validated 1 items.")

        idp_engine.converter.get_channels_from_file.side_effect = None

    def test_preprocess(self, idp_engine, mock_batch):
        logger.info(
            "[test_preprocess] ACTION: Testing ImageDataPreprocessing.preprocess"
        )

        idp_engine.converter.get_channels_from_file.return_value = [
            np.ones((10, 10)),
            np.ones((10, 10)),
            np.ones((10, 10)),
        ]
        idp_engine.pipeline.apply.return_value = mock_batch

        result = idp_engine.preprocess("fake_path.png")

        if result is None:
            logger.error("[test_preprocess] Data loss: preprocess returned None")
        assert result is not None

        if len(result) != 3:
            logger.error(
                f"[test_preprocess] Validation error: expected 3 channel results, got {len(result)}"
            )
        assert len(result) == 3

        for i, channel_batch in enumerate(result):
            if not isinstance(channel_batch, list):
                logger.error(
                    f"[test_preprocess] Type mismatch: result for channel {i} is not a list"
                )
            assert isinstance(channel_batch, list)

        logger.info(f"[test_preprocess] Validated {len(result)} items.")
