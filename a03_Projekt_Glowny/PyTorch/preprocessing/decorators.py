import logging
import math
import random
from functools import wraps
from typing import Any, Callable, Optional, TypeAlias, TypeVar, Union, cast, overload

import numpy as np

from common_utils import build_all

ClassType = TypeVar("ClassType")
FunctionType = TypeVar("FunctionType", bound=Callable[..., Any])

CT: TypeAlias = type[ClassType]
FuncDec: TypeAlias = Callable[[FunctionType], FunctionType]
ClsDec: TypeAlias = Callable[[CT], CT]

logger = logging.getLogger(__name__)


def auto_fill_color(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, *args, **kwargs):
        M = np.asanyarray(M)
        if kwargs.get("fill") is None:
            values, counts = np.unique(M, return_counts=True)
            chosen_fill = values[np.argmax(counts)]

            logger.debug(
                f"[auto_fill_color] 'fill' is None. Analyzed matrix: "
                f"found {len(values)} unique values. "
                f"Selecting dominant color as fill: {chosen_fill}"
            )

            kwargs["fill"] = chosen_fill

        return func(self, M, *args, **kwargs)

    return cast(FunctionType, wrapper)


def with_dimensions(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, *args, **kwargs):
        M = np.asanyarray(M)
        h, w = M.shape

        logger.debug(
            f"[with_dimensions] Extracted matrix dimensions: height={h}, width={w}"
        )

        return func(self, M, h, w, *args, **kwargs)

    return cast(FunctionType, wrapper)


def prepare_angle(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, *args, **kwargs):
        is_right = kwargs.get("is_right")
        limits: dict[Optional[bool], tuple[int, int]] = {
            True: (0, 30),
            False: (-30, 0),
            None: (-30, 30),
        }

        low, high = limits.get(is_right, (-50, 50))
        logger.debug(
            f"[prepare_angle] Direction is_right={is_right}. Setting sampling range to ({low}, {high})"
        )

        if kwargs.get("angle") is None:
            sampled_angle = random.randint(low, high)

            logger.debug(
                f"[prepare_angle] 'angle' is None. Randomly sampled angle: {sampled_angle}"
            )
            kwargs["angle"] = sampled_angle

        return func(self, M, *args, **kwargs)

    return cast(FunctionType, wrapper)


def prepare_values(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(
        self, M: Any, h: int, w: int, angle: float = 0, fill: int = 0, **kwargs
    ):
        rad = math.radians(angle)
        M = np.asanyarray(M)

        params = {
            "cos_a": math.cos(rad),
            "sin_a": math.sin(rad),
            "cx": w / 2.0,
            "cy": h / 2.0,
            "new_matrix": np.full((h, w), fill, dtype=M.dtype),
        }

        logger.debug(
            f"[prepare_values] Calculated rotation params for angle {angle}°: "
            f"cos={params['cos_a']:.4f}, sin={params['sin_a']:.4f}, "
            f"center=({params['cx']}, {params['cy']})"
        )

        logger.debug(
            f"[prepare_values] Initialized new matrix {h}x{w} "
            f"with fill_color={fill} and dtype={M.dtype}"
        )

        return func(self, M, h, w, params=params, angle=angle, fill=fill, **kwargs)

    return cast(FunctionType, wrapper)


def get_number_repeats(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "repeats" not in kwargs and len(args) <= 2:
            sampled_repeats = random.randrange(2, 5)

            logger.debug(
                f"[get_number_repeats] 'repeats' missing in kwargs (args_len: {len(args)}). "
                f"Randomly sampled repeats: {sampled_repeats}"
            )

            kwargs["repeats"] = sampled_repeats

        return func(*args, **kwargs)

    return cast(FunctionType, wrapper)


def kernel_data_processing(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, *args, **kwargs):
        k_size = kwargs.get("kernel_size")

        if k_size is None and args:
            k_size = args[0]
            logger.debug(
                f"[kernel_data_processing] 'kernel_size' not in kwargs, extracted from args: {k_size}"
            )

        if k_size is not None:
            r_val = (k_size - 1) // 2
            r_range = range(-r_val, r_val + 1)

            logger.debug(
                f"[kernel_data_processing] Processing for kernel_size: {k_size}. "
                f"Calculated radius r_val: {r_val}, injecting range: {list(r_range)} into kwargs"
            )
            kwargs["r"] = r_range
        else:
            logger.warning(
                "[kernel_data_processing] 'kernel_size' is None. Cannot calculate 'r' range."
            )

        return func(self, M, *args, **kwargs)

    return cast(FunctionType, wrapper)


def parameter_complement(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, matrix: Any, *args, **kwargs):
        matrix = np.asanyarray(matrix)
        h, w = matrix.shape[:2]

        if kwargs.get("auto_params"):
            initial_b_size = int(min(h, w) * 0.25)
            b_size = initial_b_size

            if b_size % 2 == 0:
                b_size += 1

            final_b_size = max(3, b_size)

            logger.debug(
                f"[parameter_complement] auto_params=True. Matrix size {w}x{h}. "
                f"Calculated block_size: initial={initial_b_size}, "
                f"after odd-check={b_size}, final(min=3)={final_b_size}. "
                f"Injected c=7 into kwargs"
            )

            kwargs["block_size"] = final_b_size
            kwargs["c"] = 7

        return func(self, matrix, *args, **kwargs)

    return cast(FunctionType, wrapper)


@overload
def apply_to_methods(decorators: list[FuncDec], method_names: list[str]) -> ClsDec: ...


@overload
def apply_to_methods(decorators: FuncDec, method_names: str) -> ClsDec: ...


@overload
def apply_to_methods(decorators: FuncDec, method_names: list[str]) -> ClsDec: ...


@overload
def apply_to_methods(decorators: list[FuncDec], method_names: str) -> ClsDec: ...


def apply_to_methods(
    decorators: Union[FuncDec, list[FuncDec]], method_names: Union[str, list[str]]
) -> ClsDec:

    decs = decorators if isinstance(decorators, list) else [decorators]
    names = method_names if isinstance(method_names, list) else [method_names]

    def class_rebuilder(cls: CT) -> CT:
        for name in names:
            current_attr = getattr(cls, name, None)

            if current_attr:
                logger.debug(
                    f"[apply_to_methods] Target method '{name}' found in class '{cls.__name__}'. "
                    f"Applying {len(decs)} decorator(s)."
                )

                if isinstance(current_attr, (classmethod, staticmethod)):
                    func = current_attr.__func__

                    for decorator in decs:
                        logger.debug(
                            f"[apply_to_methods] Wrapping {type(current_attr).__name__} '{name}' with {decorator.__name__}"
                        )
                        func = decorator(func)

                    setattr(cls, name, type(current_attr)(func))

                elif callable(current_attr):
                    for decorator in decs:
                        logger.debug(
                            f"[apply_to_methods] Wrapping instance method '{name}' with {decorator.__name__}"
                        )
                        current_attr = decorator(current_attr)

                    setattr(cls, name, current_attr)
            else:
                logger.warning(
                    f"[apply_to_methods] Method '{name}' not found in class '{cls.__name__}'. Skipping decoration."
                )

        return cls

    return class_rebuilder


__all__ = build_all(locals())
