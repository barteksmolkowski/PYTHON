import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from preprocessing import (
    apply_to_methods,
    auto_fill_color,
    get_number_repeats,
    kernel_data_processing,
    parameter_complement,
    prepare_angle,
    prepare_values,
    with_dimensions,
)

logger = logging.getLogger(__name__)


class MockBase:
    def method(self, M, *args, **kwargs):
        return M, args, kwargs


class TestDecorators:
    @pytest.fixture
    def mock_mtx(self):
        return np.array(
            [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=np.uint8
        )

    def test_auto_fill_color(self, mock_mtx):
        logger.info("ACTION: Testing auto_fill_color")
        decorated = auto_fill_color(MockBase().method)
        _, _, kwargs = decorated(None, mock_mtx)

        if kwargs.get("fill") != 1:
            logger.error(
                f"Assertion failed: fill {kwargs.get('fill')} is not dominant 1"
            )
        assert kwargs["fill"] == 1
        logger.info("SUCCESS: auto_fill_color verified")

    def test_with_dimensions(self, mock_mtx):
        logger.info("ACTION: Testing with_dimensions")
        decorated = with_dimensions(MockBase().method)
        _, args, _ = decorated(None, mock_mtx)

        if args[0] != 4 or args[1] != 4:
            logger.error(f"Assertion failed: dims {args[0]}x{args[1]} != 4x4")
        assert args[0] == 4 and args[1] == 4
        logger.info("SUCCESS: with_dimensions verified")

    def test_prepare_angle(self, mock_mtx):
        logger.info("ACTION: Testing prepare_angle")
        decorated = prepare_angle(MockBase().method)
        _, _, kwargs = decorated(None, mock_mtx, is_right=True)
        angle = kwargs.get("angle")

        if not (0 <= angle <= 30):
            logger.error(f"Assertion failed: angle {angle} outside [0, 30]")
        assert 0 <= angle <= 30
        logger.info("SUCCESS: prepare_angle verified")

    def test_prepare_values(self, mock_mtx):
        logger.info("ACTION: Testing prepare_values")
        decorated = prepare_values(MockBase().method)
        h, w, fill = 4, 4, 10
        _, _, kwargs = decorated(None, mock_mtx, h=h, w=w, angle=0, fill=fill)

        params = kwargs.get("params")
        new_m = params["new_matrix"]

        if new_m.shape != (4, 4) or new_m.dtype != np.uint8:
            logger.error("Assertion failed: physical mismatch in generated matrix")
        assert new_m.shape == (4, 4)
        assert new_m.dtype == np.uint8

        expected_sum = h * w * fill
        if np.sum(new_m) != expected_sum:
            logger.error(f"Assertion failed: sum {np.sum(new_m)} != {expected_sum}")
        assert np.sum(new_m) == expected_sum
        logger.info("SUCCESS: prepare_values verified")

    def test_get_number_repeats(self):
        logger.info("ACTION: Testing get_number_repeats")

        @get_number_repeats
        def dummy_func(*args, **kwargs):
            return kwargs

        res = dummy_func(None, "mtx")
        repeats = res.get("repeats")

        if not (2 <= repeats < 5):
            logger.error(f"Assertion failed: repeats {repeats} out of range [2, 5)")
        assert 2 <= repeats < 5
        logger.info("SUCCESS: get_number_repeats verified")

    def test_kernel_data_processing(self, mock_mtx):
        logger.info("ACTION: Testing kernel_data_processing")
        decorated = kernel_data_processing(MockBase().method)
        _, _, kwargs = decorated(None, mock_mtx, kernel_size=5)

        r_val = list(kwargs.get("r"))
        if r_val != [-2, -1, 0, 1, 2]:
            logger.error(f"Assertion failed: range {r_val} incorrect for k=5")
        assert r_val == [-2, -1, 0, 1, 2]
        logger.info("SUCCESS: kernel_data_processing verified")

    def test_parameter_complement(self, mock_mtx):
        logger.info("ACTION: Testing parameter_complement")
        decorated = parameter_complement(MockBase().method)
        _, _, kwargs = decorated(None, mock_mtx, auto_params=True)

        if kwargs.get("c") != 7 or not isinstance(kwargs.get("block_size"), int):
            logger.error("Assertion failed: auto_params complement failed")
        assert kwargs["c"] == 7
        assert kwargs["block_size"] >= 3
        logger.info("SUCCESS: parameter_complement verified")

    def test_apply_to_methods(self):
        logger.info("ACTION: Testing apply_to_methods")

        class Target:
            def action(self, M):
                return M

        mock_dec = MagicMock(side_effect=lambda x: x)
        rebuilder = apply_to_methods(mock_dec, "action")
        rebuilder(Target)

        if not mock_dec.called:
            logger.error("Assertion failed: decorator not applied to method")
        assert mock_dec.called
        logger.info("SUCCESS: apply_to_methods verified")
