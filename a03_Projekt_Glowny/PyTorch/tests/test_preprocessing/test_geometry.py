import logging
from unittest.mock import patch

import numpy as np
from preprocessing import ImageGeometry

logger = logging.getLogger("test_logger")


class TestImageGeometry:
    def test_resize(self, geometry_engine, mock_geometry_mtx):
        logger.info("[test_resize] ACTION: Testing resize")
        target_size = (28, 28)

        with patch.object(
            ImageGeometry,
            "_downscale_vectorized",
            return_value=np.ones(target_size, dtype=np.uint8),
        ):
            res = geometry_engine.resize(mock_geometry_mtx, new_size=target_size)

            if res.shape != target_size:
                logger.error(
                    f"[test_resize] Shape mismatch: {res.shape} != {target_size}"
                )
            assert res.shape == target_size

            if res.dtype != np.uint8:
                logger.error(
                    f"[test_resize] Logic error: dtype {res.dtype} is not uint8"
                )
            assert res.dtype == np.uint8

            expected_sum = 28 * 28
            if np.sum(res) != expected_sum:
                logger.error(
                    f"[test_resize] Data loss: sum {np.sum(res)} != {expected_sum}"
                )
            assert np.sum(res) == expected_sum

        logger.info("[test_resize] Validated 1 items.")

    def test_prepare_standard_geometry(self, geometry_engine, mock_geometry_mtx):
        logger.info(
            "[test_prepare_standard_geometry] ACTION: Testing prepare_standard_geometry"
        )
        target_size = (28, 28)
        padding = 2
        inner_size = (24, 24)

        with patch.object(
            ImageGeometry, "resize", return_value=np.ones(inner_size, dtype=np.uint8)
        ):
            res = geometry_engine.prepare_standard_geometry(
                mock_geometry_mtx, target_size=target_size, padding=padding, pad_value=0
            )

            if res.shape != target_size:
                logger.error(
                    f"[test_prepare_standard_geometry] Out of bounds: {res.shape} != {target_size}"
                )
            assert res.shape == target_size

            if res.dtype != np.uint8:
                logger.error(
                    f"[test_prepare_standard_geometry] Logic error: dtype {res.dtype} is not uint8"
                )
            assert res.dtype == np.uint8

            expected_sum = 24 * 24
            if np.sum(res) != expected_sum:
                logger.error(
                    f"[test_prepare_standard_geometry] Data loss: sum {np.sum(res)} != {expected_sum}"
                )
            assert np.sum(res) == expected_sum

        logger.info("[test_prepare_standard_geometry] Validated 1 items.")
