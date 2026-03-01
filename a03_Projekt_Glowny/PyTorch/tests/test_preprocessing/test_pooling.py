import logging

import numpy as np

logger = logging.getLogger("test_logger")


class TestPooling:
    def test_max_pool_with_padding(self, pooling_engine, mock_pooling_mtx):
        logger.info(
            "[test_max_pool_with_padding] ACTION: Testing max_pool with padding"
        )

        result = pooling_engine.max_pool(
            mock_pooling_mtx, kernel_size=(2, 2), stride=2, pad_width=1, pad_values=0
        )

        if result.shape != (3, 3):
            logger.error(
                f"[test_max_pool_with_padding] Shape mismatch: expected (3, 3), got {result.shape}"
            )
        assert result.shape == (3, 3)

        if result.dtype != np.uint8:
            logger.error(
                f"[test_max_pool_with_padding] Logic error: dtype {result.dtype} is not uint8"
            )
        assert result.dtype == np.uint8

        logger.info("[test_max_pool_with_padding] Validated 1 items.")

    def test_max_pool(self, pooling_engine, mock_pooling_mtx):
        logger.info("[test_max_pool] ACTION: Testing max_pool")

        kernel_size = (2, 2)
        stride = 2

        result = pooling_engine.max_pool(
            mock_pooling_mtx, kernel_size=kernel_size, stride=stride
        )

        if result.shape != (2, 2):
            logger.error(
                f"[test_max_pool] Out of bounds: expected (2, 2), got {result.shape}"
            )
        assert result.shape == (2, 2)

        if result.dtype != np.uint8:
            logger.error(
                f"[test_max_pool] Logic error: dtype {result.dtype} is not uint8"
            )
        assert result.dtype == np.uint8

        expected = np.array([[6, 8], [14, 16]], dtype=np.uint8)
        if not np.array_equal(result, expected):
            logger.error(f"[test_max_pool] Data loss: values mismatch. Got {result}")
        assert np.array_equal(result, expected)

        actual_sum = np.sum(result)
        if actual_sum != 44:
            logger.error(f"[test_max_pool] Data loss: sum {actual_sum} != 44")
        assert actual_sum == 44

        logger.info("[test_max_pool] Validated 1 items.")
