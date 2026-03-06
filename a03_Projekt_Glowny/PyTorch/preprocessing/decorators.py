import logging
import math
import random
from functools import wraps
from typing import Any, Callable, Optional, TypeAlias, TypeVar, Union, cast, overload

import numpy as np

ClassType = TypeVar("ClassType")
FunctionType = TypeVar("FunctionType", bound=Callable[..., Any])

CT: TypeAlias = type[ClassType]
FuncDec: TypeAlias = Callable[[FunctionType], FunctionType]
ClsDec: TypeAlias = Callable[[CT], CT]

logger = logging.getLogger(__name__)


def calculate_fill_color(M: np.ndarray) -> int:
    values, counts = np.unique(M, return_counts=True)
    return int(values[np.argmax(counts)])


def calculate_rotation_params(
    M: np.ndarray, h: int, w: int, angle: float, fill: int
) -> dict[str, Any]:
    rad = math.radians(angle)
    return {
        "cos_a": math.cos(rad),
        "sin_a": math.sin(rad),
        "cx": w / 2.0,
        "cy": h / 2.0,
        "new_matrix": np.full((h, w), fill, dtype=M.dtype),
    }


def get_angle_range(is_right: Optional[bool]) -> tuple[int, int]:
    limits = {True: (0, 30), False: (-30, 0), None: (-30, 30)}
    return limits.get(is_right, (-50, 50))


def calculate_block_size(h: int, w: int) -> int:
    b_size = int(min(h, w) * 0.25)
    if b_size % 2 == 0:
        b_size += 1
    return max(3, b_size)


def auto_fill_color(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, *args, **kwargs):
        M_arr = np.asanyarray(M)
        if kwargs.get("fill") is None:
            kwargs["fill"] = calculate_fill_color(M_arr)
            logger.debug(f"[auto_fill_color] Injected fill={kwargs['fill']}")
        return func(self, M_arr, *args, **kwargs)

    return cast(FunctionType, wrapper)


def with_dimensions(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, *args, **kwargs):
        M_arr = np.asanyarray(M)
        h, w = M_arr.shape[:2]
        kwargs.pop("h", None)
        kwargs.pop("w", None)
        logger.debug(f"[with_dimensions] Injected {h}x{w}")
        return func(self, M_arr, h, w, *args, **kwargs)

    return cast(FunctionType, wrapper)


def prepare_angle(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, *args, **kwargs):
        is_right = kwargs.pop("is_right", None)
        low, high = get_angle_range(is_right)
        if kwargs.get("angle") is None:
            kwargs["angle"] = random.randint(low, high)
        return func(self, M, *args, **kwargs)

    return cast(FunctionType, wrapper)


def prepare_values(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, h: int, w: int, **kwargs):
        angle = kwargs.pop("angle", 0)
        fill = kwargs.pop("fill", 0)
        params = calculate_rotation_params(np.asanyarray(M), h, w, angle, fill)
        return func(self, M, h, w, params=params, angle=angle, fill=fill, **kwargs)

    return cast(FunctionType, wrapper)


def get_number_repeats(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "repeats" not in kwargs:
            kwargs["repeats"] = random.randrange(2, 5)
        return func(*args, **kwargs)

    return cast(FunctionType, wrapper)


def kernel_data_processing(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, *args, **kwargs):
        k_size = kwargs.get("kernel_size") or (args[0] if args else None)
        if k_size is not None:
            r_val = (int(k_size) - 1) // 2
            kwargs["r"] = range(-r_val, r_val + 1)
        return func(self, M, *args, **kwargs)

    return cast(FunctionType, wrapper)


def parameter_complement(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, matrix: Any, *args, **kwargs):
        M_arr = np.asanyarray(matrix)
        if kwargs.get("auto_params"):
            h, w = M_arr.shape[:2]
            kwargs["block_size"] = calculate_block_size(h, w)
            kwargs["c"] = 7
        return func(self, M_arr, *args, **kwargs)

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
            attr = getattr(cls, name, None)
            if attr:
                target = (
                    attr.__func__
                    if isinstance(attr, (classmethod, staticmethod))
                    else attr
                )
                for dec in reversed(decs):
                    target = dec(target)
                if isinstance(attr, classmethod):
                    target = classmethod(target)
                elif isinstance(attr, staticmethod):
                    target = staticmethod(target)
                setattr(cls, name, target)
        return cls

    return class_rebuilder


__all__ = [
    "apply_to_methods",
    "auto_fill_color",
    "calculate_block_size",
    "calculate_fill_color",
    "calculate_rotation_params",
    "get_angle_range",
    "get_number_repeats",
    "kernel_data_processing",
    "parameter_complement",
    "prepare_angle",
    "prepare_values",
    "with_dimensions",
]
