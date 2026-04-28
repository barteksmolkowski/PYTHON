import logging
import math
import random
from functools import wraps
from typing import (
    Any,
    Callable,
    List,
    Literal,
    ParamSpec,
    Union,
    cast,
    overload,
)

import numpy as np

from MyTorch import ClassType, ImageGray, JsonData, Shape, T

P = ParamSpec("P")
logger = logging.getLogger(__name__)


def calculate_fill_color(M: ImageGray) -> int:
    values, counts = np.unique(M, return_counts=True)
    return int(values[np.argmax(counts)])


def calculate_rotation_params(
    M: ImageGray, h: int, w: int, angle: float, fill: int
) -> JsonData:
    rad = math.radians(angle)
    return {
        "cos_a": math.cos(rad),
        "sin_a": math.sin(rad),
        "cx": w / 2.0,
        "cy": h / 2.0,
        "new_matrix": np.full((h, w), fill, dtype=M.dtype),
    }


def get_angle_range(is_right: bool) -> Shape:
    limits = {True: (0, 30), False: (-30, 0)}
    return limits.get(is_right, (-30, 30))


def calculate_block_size(h: int, w: int) -> int:
    b_size = int(min(h, w) * 0.25)
    b_size = b_size + 1 if b_size % 2 == 0 else b_size
    return max(3, b_size)


def auto_fill_color(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if "fill" not in kwargs:
            m_input = next(
                (arg for arg in args if isinstance(arg, np.ndarray)),
                np.array([], dtype=np.float32),
            )
            kwargs["fill"] = calculate_fill_color(cast(ImageGray, m_input))
        return func(*args, **kwargs)

    return wrapper


def with_dimensions(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        m_input = next(
            (arg for arg in args if isinstance(arg, np.ndarray)),
            np.array([], dtype=np.float32),
        )
        h, w = m_input.shape[:2]
        kwargs["h"], kwargs["w"] = int(h), int(w)
        return func(*args, **kwargs)

    return wrapper


def prepare_angle(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        is_right = bool(kwargs.get("is_right", True))
        low, high = get_angle_range(is_right)
        if "angle" not in kwargs:
            kwargs["angle"] = float(random.randint(low, high))
        return func(*args, **kwargs)

    return wrapper


def prepare_values(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        h = int(cast(Any, kwargs.get("h", 0)))
        w = int(cast(Any, kwargs.get("w", 0)))
        angle = float(cast(Any, kwargs.get("angle", 0.0)))
        fill = int(cast(Any, kwargs.get("fill", 0)))

        m_input = next(
            (arg for arg in args if isinstance(arg, np.ndarray)),
            np.array([], dtype=np.float32),
        )

        params = calculate_rotation_params(cast(ImageGray, m_input), h, w, angle, fill)

        kwargs["params"] = cast(JsonData, params)
        kwargs["angle"] = angle
        kwargs["fill"] = fill

        return func(*args, **kwargs)

    return wrapper


def get_number_repeats(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if "repeats" not in kwargs:
            kwargs["repeats"] = cast(Any, random.randrange(2, 5))
        return func(*args, **kwargs)

    return wrapper


def kernel_data_processing(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        k_size = kwargs.get("kernel_size")
        if k_size is None and args:
            for arg in args:
                if isinstance(arg, int):
                    k_size = arg
                    break

        if k_size is not None:
            r_val = (int(cast(Any, k_size)) - 1) // 2
            kwargs["r"] = cast(Any, range(-r_val, r_val + 1))

        return func(*args, **kwargs)

    return wrapper


def parameter_complement(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        m_input = next(
            (arg for arg in args if isinstance(arg, np.ndarray)),
            np.array([], dtype=np.float32),
        )

        if bool(kwargs.get("auto_params", False)):
            h, w = m_input.shape[:2]
            kwargs["block_size"] = cast(Any, calculate_block_size(int(h), int(w)))
            kwargs["c"] = cast(Any, 7)

        return func(*args, **kwargs)

    return wrapper


@overload
def apply_to_methods(
    decorators: Union[Callable, List[Callable]],
    method_names: Literal["all"] = "all",
    exclude: Union[str, List[str]] = list(),
) -> Callable[[ClassType], ClassType]: ...


@overload
def apply_to_methods(
    decorators: Union[Callable, List[Callable]],
    method_names: Union[str, List[str]],
    exclude: Union[str, List[str]] = list(),
) -> Callable[[ClassType], ClassType]: ...


def apply_to_methods(
    decorators: Union[Callable, List[Callable]],
    method_names: Union[str, List[str], Literal["all"]] = "all",
    exclude: Union[str, List[str]] = list(),
) -> Callable[[ClassType], ClassType]:
    decs = decorators if isinstance(decorators, list) else [decorators]
    exclude_names = exclude if isinstance(exclude, list) else [exclude]

    def class_rebuilder(cls: ClassType) -> ClassType:
        if method_names == "all":
            target_names = [
                m
                for m in dir(cls)
                if callable(getattr(cls, m)) and not m.startswith("_")
            ]
        else:
            target_names = (
                method_names if isinstance(method_names, list) else [method_names]
            )

        for name in target_names:
            if name in exclude_names:
                continue

            attr = getattr(cls, name, None)
            if attr:
                is_static = isinstance(attr, staticmethod)
                is_class = isinstance(attr, classmethod)
                target = attr.__func__ if (is_static or is_class) else attr

                for dec in reversed(decs):
                    target = dec(target)

                if is_static:
                    target = staticmethod(target)
                elif is_class:
                    target = classmethod(target)

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
