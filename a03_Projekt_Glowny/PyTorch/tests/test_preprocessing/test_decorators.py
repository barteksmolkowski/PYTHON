import logging
from unittest.mock import MagicMock

import numpy as np
from preprocessing import (
    apply_to_methods,
    auto_fill_color,
    kernel_data_processing,
    parameter_complement,
    prepare_angle,
    prepare_values,
    with_dimensions,
)

logger = logging.getLogger("test_logger")


class TestDecorators:
    def test_prepare_angle(self, mock_base, decorator_mtx):
        logger.info("[test_prepare_angle] ACTION: Testing prepare_angle")
        decorated = prepare_angle(mock_base.method)

        _, _, kwargs = decorated(mock_base, decorator_mtx, is_right=True)
        angle = kwargs.get("angle")

        if not (0 <= angle <= 30):
            logger.error(
                f"[test_prepare_angle] Out of bounds: angle {angle} outside [0, 30]"
            )

        assert 0 <= angle <= 30
        logger.info("[test_prepare_angle] Validated 1 items.")

    def test_kernel_data_processing(self, mock_base, decorator_mtx):
        logger.info(
            "[test_kernel_data_processing] ACTION: Testing kernel_data_processing"
        )
        decorated = kernel_data_processing(mock_base.method)
        k_size = 5
        _, _, kwargs = decorated(mock_base, decorator_mtx, kernel_size=k_size)

        r_val = list(kwargs.get("r", []))
        expected_range = [-2, -1, 0, 1, 2]

        if r_val != expected_range:
            logger.error(
                f"[test_kernel_data_processing] Logic error: expected {expected_range}, got {r_val}"
            )

        assert r_val == expected_range
        logger.info("[test_kernel_data_processing] Validated 1 items.")

    def test_parameter_complement(self, mock_base, decorator_mtx):
        logger.info("[test_parameter_complement] ACTION: Testing parameter_complement")
        decorated = parameter_complement(mock_base.method)
        _, _, kwargs = decorated(mock_base, decorator_mtx, auto_params=True)

        c_val = kwargs.get("c")
        bs_val = kwargs.get("block_size")

        if c_val != 7 or bs_val is None:
            logger.error(
                f"[test_parameter_complement] Validation error: c={c_val}, block_size={bs_val}"
            )

        assert c_val == 7
        assert bs_val >= 3
        logger.info("[test_parameter_complement] Validated 1 items.")

    def test_auto_fill_color(self, mock_base):
        logger.info("[test_auto_fill_color] ACTION: Testing auto_fill_color")
        M = np.array([[0, 0, 1], [0, 0, 0]], dtype=np.uint8)
        decorated = auto_fill_color(mock_base.method)
        _, _, kwargs = decorated(mock_base, M)

        fill = kwargs.get("fill")
        if fill != 0:
            logger.error(
                f"[test_auto_fill_color] Logic error: expected fill 0, got {fill}"
            )

        assert fill == 0
        logger.info("[test_auto_fill_color] Validated 1 items.")

    def test_apply_to_methods(self):
        logger.info("[test_apply_to_methods] ACTION: Testing apply_to_methods")

        class Target:
            def action(self, M):
                return M

        mock_dec = MagicMock(side_effect=lambda x: x)
        rebuilder = apply_to_methods(mock_dec, "action")
        rebuilder(Target)

        if not mock_dec.called:
            logger.error(
                "[test_apply_to_methods] Validation error: decorator was not called"
            )

        assert mock_dec.called
        logger.info("[test_apply_to_methods] Validated 1 items.")

    def test_with_dimensions(self, mock_base, decorator_mtx):
        logger.info("[test_with_dimensions] ACTION: Testing with_dimensions")
        decorated = with_dimensions(mock_base.method)
        _, args, _ = decorated(mock_base, decorator_mtx)

        h, w = args[0], args[1]
        if h != 2 or w != 2:
            logger.error(
                f"[test_with_dimensions] Shape mismatch: expected 2x2, got {h}x{w}"
            )

        assert h == 2
        assert w == 2
        logger.info("[test_with_dimensions] Validated 1 items.")

    def test_get_number_repeats(self, dummy_repeats_func):
        logger.info("[test_get_number_repeats] ACTION: Testing get_number_repeats")

        res = dummy_repeats_func(None, "mtx")
        repeats = res.get("repeats")

        if not (2 <= repeats < 5):
            logger.error(
                f"[test_get_number_repeats] Out of bounds: repeats {repeats} not in [2, 5)"
            )

        assert 2 <= repeats < 5
        logger.info("[test_get_number_repeats] Validated 1 items.")

    def test_prepare_values(self, mock_base, mock_mtx):
        logger.info("[test_prepare_values] ACTION: Testing prepare_values")

        decorated = prepare_values(mock_base.method)
        h, w, fill = 4, 4, 10

        _, _, kwargs = decorated(mock_base, mock_mtx, h=h, w=w, angle=0, fill=fill)

        params = kwargs.get("params")
        new_m = params.get("new_matrix")

        if new_m is None or new_m.shape != (4, 4) or new_m.dtype != np.uint8:
            logger.error(
                f"[test_prepare_values] Physical mismatch: shape={getattr(new_m, 'shape', None)}, dtype={getattr(new_m, 'dtype', None)}"
            )

        assert new_m.shape == (4, 4)
        assert new_m.dtype == np.uint8

        expected_sum = h * w * fill
        actual_sum = np.sum(new_m)

        if actual_sum != expected_sum:
            logger.error(
                f"[test_prepare_values] Data loss: sum {actual_sum} != {expected_sum}"
            )

        assert actual_sum == expected_sum
        logger.info("[test_prepare_values] Validated 1 items.")
