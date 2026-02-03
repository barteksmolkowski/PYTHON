import logging

import numpy as np
import pytest

from preprocessing.augmentation import (
    DataAugmentation,
    GeometryAugmentation,
    MorphologyAugmentation,
    NoiseAugmentation,
    RandomUniformProvider,
)

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_mtx() -> np.ndarray:
    m = np.zeros((28, 28), dtype=np.uint8)
    m[10:15, 10:15] = 255
    return m


class TestGeometryAugmentation:
    def test_horizontal_flip_logic(self, mock_mtx):
        logger.info("ACTION: Testing horizontal_flip")
        result = GeometryAugmentation().horizontal_flip(mock_mtx)

        if result.shape != (28, 28) or result.dtype != np.uint8:
            logger.error(
                f"ASSERTION FAILED: Invalid shape {result.shape} or dtype {result.dtype}"
            )
        assert result.shape == (28, 28)
        assert result.dtype == np.uint8
        logger.info("SUCCESS: horizontal_flip verified")

    def test_vertical_flip_logic(self, mock_mtx):
        logger.info("ACTION: Testing vertical_flip")
        result = GeometryAugmentation().vertical_flip(mock_mtx)

        if np.sum(result) != np.sum(mock_mtx):
            logger.error("ASSERTION FAILED: Pixel sum mismatch after vertical_flip")
        assert np.sum(result) == np.sum(mock_mtx)
        logger.info("SUCCESS: vertical_flip verified")

    def test_rotate_90_logic(self, mock_mtx):
        logger.info("ACTION: Testing rotate_90")
        result = GeometryAugmentation().rotate_90(mock_mtx, is_right=True)

        if np.array_equal(result, mock_mtx):
            logger.error("ASSERTION FAILED: rotate_90 produced identical matrix")
        assert not np.array_equal(result, mock_mtx)
        logger.info("SUCCESS: rotate_90 verified")

    def test_rotate_small_angle_math(self, mock_mtx):
        logger.info("ACTION: Testing rotate_small_angle")
        params = {
            "cos_a": 0.98,
            "sin_a": 0.17,
            "cx": 14,
            "cy": 14,
            "new_matrix": np.zeros_like(mock_mtx),
        }
        result = GeometryAugmentation().rotate_small_angle(mock_mtx, 28, 28, params)

        if result.dtype != np.uint8:
            logger.error("ASSERTION FAILED: rotate_small_angle output is not uint8")
        assert result.dtype == np.uint8
        logger.info("SUCCESS: rotate_small_angle verified")

    def test_random_shift_dimensions(self, mock_mtx):
        logger.info("ACTION: Testing random_shift")
        result = GeometryAugmentation().random_shift(mock_mtx, 28, 28, fill=0)

        if result.shape != (28, 28):
            logger.error(f"ASSERTION FAILED: Expected (28,28), got {result.shape}")
        assert result.shape == (28, 28)
        logger.info("SUCCESS: random_shift verified")


class TestNoiseAugmentation:
    def test_gaussian_noise_application(self, mock_mtx):
        logger.info("ACTION: Testing gaussian_noise")
        result = NoiseAugmentation().gaussian_noise(mock_mtx, std=20.0)

        if np.array_equal(result, mock_mtx):
            logger.error("ASSERTION FAILED: gaussian_noise did not modify pixels")
        assert not np.array_equal(result, mock_mtx)
        logger.info("SUCCESS: gaussian_noise verified")

    def test_salt_and_pepper_values(self, mock_mtx):
        logger.info("ACTION: Testing salt_and_pepper")
        result = NoiseAugmentation().salt_and_pepper(mock_mtx, prob=0.1)

        if not (np.any(result == 0) and np.any(result == 255)):
            logger.error("ASSERTION FAILED: salt_and_pepper failed to inject extremes")
        assert np.any(result == 0) and np.any(result == 255)
        logger.info("SUCCESS: salt_and_pepper verified")


class TestMorphologyAugmentation:
    def test_dilate_expansion(self, mock_mtx):
        logger.info("ACTION: Testing dilate")
        result = MorphologyAugmentation().dilate(mock_mtx, kernel_size=3)

        if np.sum(result) <= np.sum(mock_mtx):
            logger.error("ASSERTION FAILED: Dilation did not increase pixel sum")
        assert np.sum(result) > np.sum(mock_mtx)
        logger.info("SUCCESS: dilate verified")

    def test_erode_contraction(self, mock_mtx):
        logger.info("ACTION: Testing erode")
        result = MorphologyAugmentation().erode(mock_mtx, kernel_size=3)

        if np.sum(result) >= np.sum(mock_mtx):
            logger.error("ASSERTION FAILED: Erosion did not decrease pixel sum")
        assert np.sum(result) < np.sum(mock_mtx)
        logger.info("SUCCESS: erode verified")

    def test_get_boundaries_detection(self, mock_mtx):
        logger.info("ACTION: Testing get_boundaries")
        result = MorphologyAugmentation().get_boundaries(mock_mtx, kernel_size=2)

        if np.sum(result) >= np.sum(mock_mtx):
            logger.error(
                "ASSERTION FAILED: Boundaries sum should be lower than original"
            )
        assert np.sum(result) < np.sum(mock_mtx)
        logger.info("SUCCESS: get_boundaries verified")

    def test_morphology_filter_logic(self, mock_mtx):
        logger.info("ACTION: Testing morphology_filter")
        result = MorphologyAugmentation().morphology_filter(mock_mtx, 1, mode="open")

        if result.shape != (28, 28):
            logger.error("ASSERTION FAILED: morphology_filter changed matrix shape")
        assert result.shape == (28, 28)
        logger.info("SUCCESS: morphology_filter verified")


class TestDataAugmentation:
    def test_augment_output_structure(self, mock_mtx):
        logger.info("ACTION: Testing augment")
        result = DataAugmentation().augment(mock_mtx, repeats=2)

        if not isinstance(result, list):
            logger.error(f"ASSERTION FAILED: Expected list, got {type(result)}")
        assert isinstance(result, list)
        logger.info("SUCCESS: augment verified")


class TestRandomUniformProvider:
    def test_get_value_range(self):
        logger.info("ACTION: Testing get_value")
        provider = RandomUniformProvider()
        provider.low, provider.high = 0.5, 1.5
        val = provider.get_value()

        if not (0.5 <= val <= 1.5):
            logger.error(f"ASSERTION FAILED: Value {val} out of bounds [0.5, 1.5]")
        assert 0.5 <= val <= 1.5
        logger.info("SUCCESS: get_value verified")
