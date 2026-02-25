import logging
import math
import random
from functools import wraps
from typing import Any, Callable, TypeAlias, TypeVar, Union, cast, overload

import numpy as np

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
            chosen_fill = int(values[np.argmax(counts)])
            kwargs["fill"] = chosen_fill

            logger.debug(
                f"[auto_fill_color] Missing 'fill' parameter. "
                f"calculated_fill={chosen_fill} from unique_values={len(values)}"
            )
        else:
            logger.debug(f"[auto_fill_color] Using provided fill={kwargs.get('fill')}")

        return func(self, M, *args, **kwargs)

    return cast(FunctionType, wrapper)


def with_dimensions(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, *args, **kwargs):
        M = np.asanyarray(M)
        h_val, w_val = M.shape[:2]

        logger.debug(f"[with_dimensions] Parameters detected: h={h_val}, w={w_val}")

        if "h" in kwargs or "w" in kwargs:
            logger.warning(
                f"[with_dimensions] Overriding manual parameters: "
                f"h={kwargs.get('h')}, w={kwargs.get('w')} with detected_shape=({h_val}, {w_val})"
            )
            kwargs.pop("h", None)
            kwargs.pop("w", None)

        result = func(self, M, h_val, w_val, *args, **kwargs)
        logger.info(
            f"[with_dimensions] Validated dimension injection for {h_val}x{w_val} matrix."
        )
        return result

    return cast(FunctionType, wrapper)


def prepare_angle(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, *args, **kwargs):
        is_right = kwargs.pop("is_right", None)

        limits = {True: (0, 30), False: (-30, 0), None: (-30, 30)}
        low, high = limits.get(is_right, (-50, 50))

        if is_right not in limits:
            logger.warning(
                f"[prepare_angle] Unexpected value is_right='{is_right}'. Fallback range set to ({low}, {high})"
            )
        else:
            logger.debug(
                f"[prepare_angle] Selected range=({low}, {high}) for is_right={is_right}"
            )

        if kwargs.get("angle") is None:
            kwargs["angle"] = random.randint(low, high)
            logger.debug(
                f"[prepare_angle] Randomly sampled angle={kwargs['angle']} from range=({low}, {high})"
            )
        else:
            logger.debug(f"[prepare_angle] Using provided angle={kwargs['angle']}")

        return func(self, M, *args, **kwargs)

    return cast(FunctionType, wrapper)


def prepare_values(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(self, M: Any, h: int, w: int, **kwargs):
        angle = kwargs.pop("angle", 0)
        fill = kwargs.pop("fill", 0)

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
            f"[prepare_values] Transformation context prepared: angle={angle}, rad={rad:.4f}, "
            f"center=({params['cx']}, {params['cy']}), fill={fill}"
        )

        result = func(self, M, h, w, params=params, angle=angle, fill=fill, **kwargs)

        logger.info(
            f"[prepare_values] Validated parameter injection for {h}x{w} matrix."
        )
        return result

    return cast(FunctionType, wrapper)


def get_number_repeats(func: FunctionType) -> FunctionType:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "repeats" not in kwargs and len(args) <= 2:
            sampled_repeats = random.randrange(2, 5)

            logger.debug(
                f"[get_number_repeats] 'repeats' missing in kwargs (args_len={len(args)}). "
                f"sampled_repeats={sampled_repeats}"
            )

            kwargs["repeats"] = sampled_repeats

        result = func(*args, **kwargs)
        logger.info(
            f"[get_number_repeats] Validated repeat injection: repeats={kwargs.get('repeats')}"
        )
        return result

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
                f"[kernel_data_processing] Processing for kernel_size={k_size}. "
                f"calculated_radius={r_val}, range_len={len(r_range)}"
            )
            kwargs["r"] = r_range
        else:
            logger.warning(
                f"[kernel_data_processing] Logic error: 'kernel_size' is {k_size}. Cannot calculate 'r' range."
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
                f"[parameter_complement] auto_params=True. Matrix size={w}x{h}. "
                f"Calculated block_size: initial={initial_b_size}, "
                f"odd_corrected={b_size}, final_min_3={final_b_size}. "
                f"Injected c=7"
            )

            kwargs["block_size"] = final_b_size
            kwargs["c"] = 7
        else:
            logger.debug(
                f"[parameter_complement] auto_params=False. Using manual parameters: {kwargs}"
            )

        result = func(self, matrix, *args, **kwargs)
        logger.info(
            f"[parameter_complement] Validated parameter injection for {h}x{w} matrix."
        )
        return result

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
                    f"Applying decs_count={len(decs)}."
                )

                if isinstance(current_attr, (classmethod, staticmethod)):
                    func = current_attr.__func__
                    for decorator in decs:
                        dec_name = getattr(decorator, "__name__", "unknown")
                        logger.debug(
                            f"[apply_to_methods] Wrapping {type(current_attr).__name__} '{name}' with decorator='{dec_name}'"
                        )
                        func = decorator(func)
                    setattr(cls, name, type(current_attr)(func))

                elif callable(current_attr):
                    wrapped_func = current_attr
                    for decorator in decs:
                        dec_name = getattr(decorator, "__name__", "unknown")
                        logger.debug(
                            f"[apply_to_methods] Wrapping instance method '{name}' with decorator='{dec_name}'"
                        )
                        wrapped_func = decorator(wrapped_func)

                    setattr(cls, name, wrapped_func)
            else:
                logger.error(
                    f"[apply_to_methods] Method mapping error: '{name}' not found in class '{cls.__name__}'"
                )

        logger.info(
            f"[apply_to_methods] Validated decoration for class '{cls.__name__}' (methods_count={len(names)})."
        )
        return cls

    return class_rebuilder


__all__ = [
    "apply_to_methods",
    "auto_fill_color",
    "get_number_repeats",
    "kernel_data_processing",
    "parameter_complement",
    "prepare_angle",
    "prepare_values",
    "with_dimensions",
]
