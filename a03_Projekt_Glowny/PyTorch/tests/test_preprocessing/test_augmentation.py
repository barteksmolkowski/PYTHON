import logging

import numpy as np

logger = logging.getLogger(__name__)


class TestGeometryAugmentation:
    def test_vertical_flip_logic(self, ga_engine, mock_mtx):
        logger.info("[test_vertical_flip_logic] Testing vertical_flip")
        result = ga_engine.vertical_flip(mock_mtx)

        sum_res, sum_orig = np.sum(result), np.sum(mock_mtx)
        if sum_res != sum_orig:
            logger.error(
                f"[test_vertical_flip_logic] Pixel sum mismatch: {sum_res} vs {sum_orig}"
            )

        assert sum_res == sum_orig
        logger.info("[test_vertical_flip_logic] Validated 1 items.")

    def test_rotate_90_logic(self, ga_engine, mock_mtx):
        logger.info("[test_rotate_90_logic] Testing rotate_90")
        result = ga_engine.rotate_90(mock_mtx, is_right=True)

        if np.array_equal(result, mock_mtx):
            logger.error(
                "[test_rotate_90_logic] Data loss: rotate_90 produced identical matrix"
            )

        assert not np.array_equal(result, mock_mtx)
        logger.info("[test_rotate_90_logic] Validated 1 items.")

    def test_horizontal_flip_logic(self, ga_engine, mock_mtx):
        logger.info("[test_horizontal_flip_logic] Testing horizontal_flip")
        result = ga_engine.horizontal_flip(mock_mtx)

        if result.shape != (28, 28) or result.dtype != np.uint8:
            logger.error(
                f"[test_horizontal_flip_logic] Invalid output: shape={result.shape}, dtype={result.dtype}"
            )

        assert result.shape == (28, 28)
        assert result.dtype == np.uint8
        logger.info("[test_horizontal_flip_logic] Validated 1 items.")

    def test_random_shift_dimensions(self, ga_engine, mock_mtx):
        logger.info(
            "[test_random_shift_dimensions] Testing random_shift [Dimensions & Fill Control]"
        )
        h, w = 28, 28

        result = ga_engine.random_shift.__wrapped__(
            ga_engine, M=mock_mtx, h=h, w=w, fill=0
        )

        if result.shape != (h, w):
            logger.error(
                f"[test_random_shift_dimensions] Shape mismatch: expected {(h, w)}, got {result.shape}"
            )

        assert result.shape == (h, w)
        assert result.dtype == np.uint8
        logger.info("[test_random_shift_dimensions] Validated 1 items.")

    def test_rotate_small_angle_math(self, ga_engine, mock_mtx):
        logger.info(
            "[test_rotate_small_angle_math] Testing rotate_small_angle via raw logic"
        )

        raw_rotate = ga_engine.rotate_small_angle
        while hasattr(raw_rotate, "__wrapped__"):
            raw_rotate = raw_rotate.__wrapped__

        result = raw_rotate(
            ga_engine,
            mock_mtx,
            28,
            28,
            params={
                "cos_a": 0.98,
                "sin_a": 0.17,
                "cx": 14,
                "cy": 14,
                "new_matrix": np.zeros_like(mock_mtx),
            },
            angle=0.0,
            fill=0,
        )

        if result.dtype != np.uint8 or result.shape != mock_mtx.shape:
            logger.error(
                f"[test_rotate_small_angle_math] Validation failed: shape={result.shape}, dtype={result.dtype}"
            )

        assert result.dtype == np.uint8
        assert result.shape == mock_mtx.shape
        logger.info("[test_rotate_small_angle_math] Validated 1 items.")


class TestNoiseAugmentation:
    def test_gaussian_noise_application(self, noise_engine, mock_mtx):
        logger.info("[test_gaussian_noise_application] ACTION: Testing gaussian_noise")
        result = noise_engine.gaussian_noise(mock_mtx, std=20.0)

        if np.array_equal(result, mock_mtx):
            logger.error(
                "[test_gaussian_noise_application] Data loss: gaussian_noise did not modify pixels"
            )

        assert not np.array_equal(result, mock_mtx)
        logger.info("[test_gaussian_noise_application] Validated 1 items.")

    def test_salt_and_pepper_values(self, noise_engine, mock_mtx):
        logger.info("[test_salt_and_pepper_values] ACTION: Testing salt_and_pepper")
        result = noise_engine.salt_and_pepper(mock_mtx, prob=0.1)

        has_min = np.any(result == 0)
        has_max = np.any(result == 255)

        if not (has_min and has_max):
            logger.error(
                f"[test_salt_and_pepper_values] Validation error: inject failed (min={has_min}, max={has_max})"
            )

        assert has_min and has_max
        logger.info("[test_salt_and_pepper_values] Validated 1 items.")


class TestMorphologyAugmentation:
    def test_dilate_expansion(self, morph_engine, mock_mtx):
        logger.info("[test_dilate_expansion] ACTION: Testing dilate")
        result = morph_engine.dilate(mock_mtx, kernel_size=3)

        sum_orig = np.sum(mock_mtx)
        sum_res = np.sum(result)

        if sum_res <= sum_orig:
            logger.error(
                f"[test_dilate_expansion] Logic error: Dilation failed to expand. {sum_res} <= {sum_orig}"
            )

        assert sum_res > sum_orig
        logger.info("[test_dilate_expansion] Validated 1 items.")

    def test_erode_contraction(self, morph_engine, mock_mtx):
        logger.info("[test_erode_contraction] ACTION: Testing erode")
        result = morph_engine.erode(mock_mtx, kernel_size=3)

        sum_orig = np.sum(mock_mtx)
        sum_res = np.sum(result)

        if sum_res >= sum_orig:
            logger.error(
                f"[test_erode_contraction] Logic error: Erosion failed to contract. {sum_res} >= {sum_orig}"
            )

        assert sum_res < sum_orig
        logger.info("[test_erode_contraction] Validated 1 items.")

    def test_morphology_filter_logic(self, morph_engine, mock_mtx):
        logger.info("[test_morphology_filter_logic] ACTION: Testing morphology_filter")
        result = morph_engine.morphology_filter(mock_mtx, 1, mode="open")

        if result.shape != (28, 28):
            logger.error(
                f"[test_morphology_filter_logic] Out of bounds: Expected (28, 28), got {result.shape}"
            )

        assert result.shape == (28, 28)
        logger.info("[test_morphology_filter_logic] Validated 1 items.")

    def test_get_boundaries_detection(self, morph_engine, mock_mtx):
        logger.info("[test_get_boundaries_detection] ACTION: Testing get_boundaries")
        result = morph_engine.get_boundaries(mock_mtx, kernel_size=2)

        sum_orig = np.sum(mock_mtx)
        sum_bound = np.sum(result)

        if sum_bound >= sum_orig:
            logger.error(
                f"[test_get_boundaries_detection] Logic error: Boundary sum {sum_bound} is not lower than {sum_orig}"
            )

        assert sum_bound < sum_orig
        logger.info("[test_get_boundaries_detection] Validated 1 items.")


class TestDataAugmentation:
    def test_augment_output_structure(self, data_aug_engine, mock_mtx):
        logger.info("[test_augment_output_structure] ACTION: Testing augment")
        repeats = 2
        result = data_aug_engine.augment(mock_mtx, repeats=repeats)

        if not isinstance(result, list):
            logger.error(
                f"[test_augment_output_structure] Validation error: Expected list, got {type(result)}"
            )

        assert isinstance(result, list)

        if len(result) == 0:
            logger.warning(
                f"[test_augment_output_structure] Data loss: Augmentation returned 0 samples for repeats={repeats}"
            )

        logger.info(f"[test_augment_output_structure] Validated {len(result)} items.")


class TestRandomUniformProvider:
    def test_get_value_range(self, uniform_provider):
        logger.info(
            "[test_get_value_range] ACTION: Testing get_value range and precision"
        )

        low_val, high_val = 0.5, 1.5
        uniform_provider.low, uniform_provider.high = low_val, high_val

        val = uniform_provider.get_value()

        is_in_range = (
            (low_val <= val <= high_val)
            or np.isclose(val, low_val)
            or np.isclose(val, high_val)
        )

        if not is_in_range:
            logger.error(
                f"[test_get_value_range] Out of bounds: Value {val:.4f} is strictly outside [{low_val}, {high_val}]"
            )

        assert is_in_range
        assert isinstance(val, float)
        logger.info(f"[test_get_value_range] Validated 1 items with value={val:.4f}")
