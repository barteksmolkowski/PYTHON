import logging

import numpy as np

logger = logging.getLogger("test_logger")


class TestNormalization:
    def test_normalize_zero_denominator(self, norm_engine):
        logger.info(
            "[test_normalize_zero_denominator] ACTION: Testing normalize zero denominator edge case"
        )
        mtx = np.ones((2, 2))

        res = norm_engine.normalize(mtx, old_r=(10, 10), new_r=(0, 1))

        if not np.all(res == 0):
            logger.error(
                f"[test_normalize_zero_denominator] Logic error: Expected all 0, got {res}"
            )

        assert np.all(res == 0)
        logger.info("[test_normalize_zero_denominator] Validated 1 items.")

    def test_normalize(self, norm_engine, mock_norm_mtx):
        logger.info("[test_normalize] ACTION: Testing normalize")
        res = norm_engine.normalize(mock_norm_mtx, old_r=(0, 100), new_r=(0, 1))

        if res.shape != (2, 2):
            logger.error(f"[test_normalize] Out of bounds: shape {res.shape} mismatch")
        assert res.shape == (2, 2)

        actual_sum = np.sum(res)
        if not np.isclose(actual_sum, 1.0):
            logger.error(f"[test_normalize] Data loss: sum {actual_sum} != 1.0")

        assert np.isclose(actual_sum, 1.0)
        logger.info("[test_normalize] Validated 1 items.")

    def test_z_score_normalization(self, norm_engine, mock_norm_mtx):
        logger.info(
            "[test_z_score_normalization] ACTION: Testing z_score_normalization"
        )
        res = norm_engine.z_score_normalization(mock_norm_mtx)

        if res.shape != (2, 2):
            logger.error(f"[test_z_score_normalization] Out of bounds: shape mismatch")
        assert res.shape == (2, 2)

        mean_val = np.mean(res)
        if not np.isclose(mean_val, 0.0, atol=1e-7):
            logger.error(
                f"[test_z_score_normalization] Logic error: mean {mean_val} != 0"
            )
        assert np.isclose(mean_val, 0.0, atol=1e-7)

        std_val = np.std(res)
        if not np.isclose(std_val, 1.0, atol=1e-7):
            logger.error(
                f"[test_z_score_normalization] Logic error: std {std_val} != 1"
            )
        assert np.isclose(std_val, 1.0, atol=1e-7)

        logger.info("[test_z_score_normalization] Validated 1 items.")

    def test_process(self, norm_engine, mock_norm_mtx):
        logger.info("[test_process] ACTION: Testing process")
        res_minmax = norm_engine.process(
            mock_norm_mtx, use_z_score=False, old_r=(0, 40), new_r=(0, 1)
        )

        if res_minmax.shape != (2, 2):
            logger.error(
                f"[test_process] Out of bounds: shape mismatch {res_minmax.shape}"
            )
        assert res_minmax.shape == (2, 2)

        max_val = res_minmax[1, 1]
        if max_val != 1.0:
            logger.error(f"[test_process] Data loss: max value {max_val} != 1.0")

        assert max_val == 1.0
        logger.info("[test_process] Validated 1 items.")
