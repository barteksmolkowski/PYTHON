import logging
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    overload,
)

import matplotlib.pyplot as plt
import numpy as np
from common_utils import class_autologger
from matplotlib.axes import Axes
from numpy.lib.stride_tricks import sliding_window_view

from .decorators import (
    apply_to_methods,
    auto_fill_color,
    get_number_repeats,
    kernel_data_processing,
    prepare_angle,
    prepare_values,
    with_dimensions,
)

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]

OptBool: TypeAlias = Optional[bool]
OptFloat: TypeAlias = Optional[float]

Kernel: TypeAlias = int


class DataAugmentationProtocol(Protocol):
    def augment(
        self, M: Mtx, repeats: Optional[int] = None, debug: bool = False
    ) -> MtxList: ...


class GeometryAugmentationProtocol(Protocol):
    def horizontal_flip(self, M: Mtx) -> Mtx: ...
    def vertical_flip(self, M: Mtx) -> Mtx: ...

    @overload
    def rotate_90(self, M: Mtx, is_right: Literal[True] = True) -> Mtx: ...
    @overload
    def rotate_90(self, M: Mtx, is_right: Literal[False]) -> Mtx: ...
    def rotate_90(self, M: Mtx, is_right: bool = True) -> Mtx: ...

    def rotate_small_angle(
        self,
        M: Mtx,
        h: int,
        w: int,
        params: dict,
        is_right: OptBool = None,
        angle: OptFloat = None,
        fill: Optional[int] = None,
        **kwargs,
    ) -> Mtx: ...

    def random_shift(
        self,
        M: Mtx,
        h: int,
        w: int,
        max_dx_dy: Optional[tuple[int, int]] = None,
        fill: Optional[int] = None,
        is_right: OptBool = None,
        **kwargs,
    ) -> Mtx: ...


class NoiseAugmentationProtocol(Protocol):
    def gaussian_noise(self, M: Mtx, std: OptFloat = None) -> Mtx: ...
    def salt_and_pepper(self, M: Mtx, prob: OptFloat = None) -> Mtx: ...


class MorphologyAugmentationProtocol(Protocol):
    def dilate(self, M: Mtx, kernel_size: Kernel, **kwargs: Any) -> Mtx: ...
    def erode(
        self, M: Mtx, kernel_size: Kernel, fill: int = 0, **kwargs: Any
    ) -> Mtx: ...
    def get_boundaries(
        self, M: Mtx, kernel_size: Kernel, fill: int = 0, **kwargs: Any
    ) -> Mtx: ...
    def morphology_filter(
        self,
        M: Mtx,
        kernel_size: Kernel,
        fill: int = 0,
        mode: Literal["open", "close"] = "open",
        **kwargs: Any,
    ) -> Mtx: ...


class ParameterProviderProtocol(Protocol):
    def get_value(self) -> float: ...


def _pick_random_pipeline(
    available_keys: List[str],
    groups: Dict[str, List[Tuple[Callable, str]]],
    num_samples: int = 3,
) -> Tuple[List[Tuple[Callable, str]], List[str]]:
    import random

    selected_keys = random.sample(available_keys, min(len(available_keys), num_samples))
    pipeline = [random.choice(groups[k]) for k in selected_keys]
    return pipeline, selected_keys


def _get_augmentation_groups(
    geometry: GeometryAugmentationProtocol,
    noise: NoiseAugmentationProtocol,
    morphology: MorphologyAugmentationProtocol,
) -> Dict[str, List[Tuple[Callable, str]]]:
    return {
        "geometry": [
            (getattr(geometry, m), m) for m in dir(geometry) if not m.startswith("_")
        ],
        "noise": [(getattr(noise, m), m) for m in dir(noise) if not m.startswith("_")],
        "morphology": [
            (getattr(morphology, m), m)
            for m in dir(morphology)
            if not m.startswith("_")
        ],
    }


def augment_engine(
    m_arr: np.ndarray,
    repeats: int,
    max_attempts: int,
    pipeline_provider: Callable[[], List[Tuple[Callable, str]]],
    orig_px: int,
    min_area_ratio: float = 0.5,
) -> Tuple[List[np.ndarray], List[List[str]], int]:
    results: List[np.ndarray] = []
    histories: List[List[str]] = []
    attempts = 0

    while len(results) < repeats and attempts < max_attempts:
        attempts += 1
        pipeline = pipeline_provider()
        current_m = m_arr.copy()
        current_history = [name for _, name in pipeline]

        for func, _ in pipeline:
            current_m = func(current_m)

        current_px = int(np.sum(current_m > 0))
        if current_px >= (orig_px * min_area_ratio):
            results.append(current_m)
            histories.append(current_history)

    return results, histories, attempts


def _calculate_grid_size(items_count: int, cols: int = 10) -> int:
    return (items_count + cols - 1) // cols


def _parse_input_indices(raw_input: str) -> List[int]:
    return [int(x.strip()) - 1 for x in raw_input.split(",") if x.strip()]


def _get_trace_report(idx: int, history: List[str]) -> str:
    steps = " -> ".join([f"[{i + 1}] {name}" for i, name in enumerate(history)])
    return f"\n[PLOT {idx + 1}] Operation Trace:\n  {steps}"


def horizontal_flip(M: np.ndarray) -> np.ndarray:
    return np.asanyarray(M)[:, ::-1]


def vertical_flip(M: np.ndarray) -> np.ndarray:
    return np.asanyarray(M)[::-1, :]


def rotate_90(M: np.ndarray, is_right: bool = True) -> np.ndarray:
    k = -1 if is_right else 1
    return np.rot90(np.asanyarray(M), k=k).copy()


def _create_supplement_all_sides(
    M: np.ndarray, dx: int, dy: int, shade_gray_color: int
) -> np.ndarray:
    return np.pad(
        M,
        pad_width=((abs(dy), abs(dy)), (abs(dx), abs(dx))),
        mode="constant",
        constant_values=shade_gray_color,
    ).astype(np.uint8)


def rotate_small_angle(
    M: np.ndarray,
    h: int,
    w: int,
    cos_a: float,
    sin_a: float,
    cx: float,
    cy: float,
    new_m: np.ndarray,
) -> np.ndarray:
    y_new, x_new = np.indices((h, w))
    tx, ty = x_new - cx, y_new - cy
    xf = tx * cos_a + ty * sin_a + cx
    yf = -tx * sin_a + ty * cos_a + cy

    mask = (xf >= 0) & (xf < w - 1) & (yf >= 0) & (yf < h - 1)
    xi, yi = xf[mask].astype(int), yf[mask].astype(int)

    new_m[y_new[mask], x_new[mask]] = M[yi, xi]
    return new_m.astype(np.uint8)


def random_shift_engine(
    M: np.ndarray, h: int, w: int, dx: int, dy: int, sx: int, sy: int, fill: int
) -> np.ndarray:
    padded = np.pad(
        M,
        pad_width=((abs(dy), abs(dy)), (abs(dx), abs(dx))),
        mode="constant",
        constant_values=fill,
    ).astype(np.uint8)

    shifted = padded[sy : sy + h, sx : sx + w].copy()
    return shifted


def get_uniform_value(low: float, high: float) -> float:
    return random.uniform(low, high)


def gaussian_noise(M: np.ndarray, std: float, noise_map: np.ndarray) -> np.ndarray:
    np_M = M.astype(np.float32)
    return np.clip(np_M + (noise_map * std), 0, 255).astype(np.uint8)


def salt_and_pepper(M: np.ndarray, prob: float, random_map: np.ndarray) -> np.ndarray:
    result = M.copy().astype(np.uint8)
    salt_threshold = 1.0 - prob / 2.0
    pepper_threshold = prob / 2.0
    result[random_map < pepper_threshold] = 0
    result[random_map > salt_threshold] = 255
    return result


def dilate(
    M: np.ndarray, kernel_size: int, sliding_window_func: Callable
) -> np.ndarray:
    return sliding_window_func(M, kernel_size, pad_value=0)


def erode(
    M: np.ndarray, kernel_size: int, fill: int, sliding_window_func: Callable
) -> np.ndarray:
    return sliding_window_func(
        M,
        kernel_size,
        op_func=lambda win, axes: np.min(win, axis=axes),
        pad_value=fill,
    )


def get_boundaries(M: np.ndarray, eroded: np.ndarray) -> np.ndarray:
    diff = M.astype(np.int16) - eroded.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8)


def morphology_filter(
    M: np.ndarray,
    kernel_size: int,
    fill: int,
    mode: str,
    dilate_func: Callable,
    erode_func: Callable,
) -> np.ndarray:
    if mode == "open":
        return dilate_func(erode_func(M, kernel_size, fill), kernel_size)
    return erode_func(dilate_func(M, kernel_size), kernel_size, fill)


def sliding_window_engine(
    M_arr: np.ndarray,
    kernel_size: int,
    pad_before: int,
    pad_after: int,
    pad_value: int,
    op_func: Callable[[np.ndarray, Tuple[int, ...]], np.ndarray],
) -> np.ndarray:
    padded = np.pad(
        M_arr,
        pad_width=((pad_before, pad_after), (pad_before, pad_after)),
        mode="constant",
        constant_values=pad_value,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernel_size, kernel_size)
    )
    result = op_func(windows, (2, 3)).astype(np.uint8)
    return result[: M_arr.shape[0], : M_arr.shape[1]]


@dataclass
@class_autologger
class DataAugmentation:
    geometry: Optional[GeometryAugmentationProtocol] = None
    noise: Optional[NoiseAugmentationProtocol] = None
    morphology: Optional[MorphologyAugmentationProtocol] = None

    logger: logging.Logger = field(init=False)
    _cached_groups: Dict[str, List[Tuple[Callable, str]]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.geometry is None:
            self.geometry = GeometryAugmentation()
        if self.noise is None:
            self.noise = NoiseAugmentation()
        if self.morphology is None:
            self.morphology = MorphologyAugmentation()

        assert self.geometry is not None
        assert self.noise is not None
        assert self.morphology is not None

        self._cached_groups = _get_augmentation_groups(
            self.geometry, self.noise, self.morphology
        )
        self.logger.info(f"Initialized with {len(self._cached_groups)} groups.")

    def _get_augmentation_groups(self) -> Dict[str, List[Tuple[Callable, str]]]:
        assert self.geometry is not None
        assert self.noise is not None
        assert self.morphology is not None

        self.logger.debug(
            f"[_get_augmentation_groups] Building groups with: "
            f"geometry={type(self.geometry).__name__}, "
            f"noise={type(self.noise).__name__}, "
            f"morphology={type(self.morphology).__name__}"
        )

        groups = _get_augmentation_groups(self.geometry, self.noise, self.morphology)

        self.logger.info(
            f"[_get_augmentation_groups] Built and silenced {len(groups)} groups."
        )
        return groups

    def _pick_random_pipeline(self) -> List[Tuple[Callable, str]]:
        available_keys = list(self._cached_groups.keys())
        pipeline, selected_keys = _pick_random_pipeline(
            available_keys, self._cached_groups
        )

        for (_, name), key in zip(pipeline, selected_keys):
            self.logger.debug(f"Selected '{name}' from group '{key}'")

        return pipeline

    @get_number_repeats
    def augment(
        self, M: Mtx, repeats: Optional[int] = None, debug: bool = False
    ) -> MtxList:
        num_repeats: int = repeats if repeats is not None else 1
        m_arr = np.asanyarray(M).astype(np.uint8)
        orig_px = int(np.sum(m_arr > 0))
        max_attempts = num_repeats * 100

        self.logger.debug(f"Starting augmentation: repeats={num_repeats}")

        results, histories, attempts = augment_engine(
            m_arr=m_arr,
            repeats=num_repeats,
            max_attempts=max_attempts,
            pipeline_provider=self._pick_random_pipeline,
            orig_px=orig_px,
        )

        if attempts >= max_attempts:
            self.logger.warning(f"Max attempts reached: {attempts}/{max_attempts}")

        if debug:
            self._display_debug_plots(results, histories)

        self.logger.info(f"Generated {len(results)} samples.")
        return results

    def _display_debug_plots(
        self, result: List[np.ndarray], histories: List[List[str]]
    ) -> None:
        cols: int = 10
        rows: int = _calculate_grid_size(len(result), cols)

        self.logger.debug(f"Grid setup: {rows}x{cols} for {len(result)} items.")

        print("\n" + "=" * 50 + " INTERACTIVE DEBUG MODE (2026) " + "=" * 50)

        while True:
            cmd: str = input("\nEnter indices (1, 5...) or 'q': ").lower()
            if cmd == "q":
                break

            try:
                indices: List[int] = _parse_input_indices(cmd)
                for idx in indices:
                    if 0 <= idx < len(histories):
                        report: str = _get_trace_report(idx, histories[idx])
                        print(report)
                    else:
                        self.logger.warning(f"Index {idx + 1} out of bounds.")
            except ValueError:
                self.logger.error(f"Invalid debug input: {cmd}")

        self.logger.info("Debug session closed.")


@class_autologger
@apply_to_methods(
    [auto_fill_color, with_dimensions], ["rotate_small_angle", "random_shift"]
)
class GeometryAugmentation:
    logger: logging.Logger = field(init=False, repr=False)

    def horizontal_flip(self, M: np.ndarray) -> np.ndarray:
        self.logger.debug(f"[horizontal_flip] Shape: {M.shape}")
        return horizontal_flip(M)

    def vertical_flip(self, M: np.ndarray) -> np.ndarray:
        self.logger.debug(f"[vertical_flip] Shape: {M.shape}")
        return vertical_flip(M)

    def rotate_90(self, M: np.ndarray, is_right: bool = True) -> np.ndarray:
        self.logger.debug(f"[rotate_90] is_right={is_right}")
        return rotate_90(M, is_right)

    @prepare_angle
    @prepare_values
    def rotate_small_angle(
        self,
        M: np.ndarray,
        h: int,
        w: int,
        params: Dict[str, Any],
        angle: float = 0.0,
        fill: int = 0,
        **kwargs,
    ) -> np.ndarray:
        self.logger.debug(f"[rotate_small_angle] angle={angle}")

        cos_a: float = float(params["cos_a"])
        sin_a: float = float(params["sin_a"])
        cx: float = float(params["cx"])
        cy: float = float(params["cy"])
        new_matrix: np.ndarray = params["new_matrix"]

        result = rotate_small_angle(
            M=M,
            h=int(h),
            w=int(w),
            cos_a=cos_a,
            sin_a=sin_a,
            cx=cx,
            cy=cy,
            new_m=new_matrix,
        )

        self.logger.info(f"[rotate_small_angle] Completed {h}x{w}")
        return result

    def _create_supplement_all_sides(
        self, M: np.ndarray, x_y_axis: Tuple[int, int], shade_gray_color: int
    ) -> np.ndarray:
        dx, dy = x_y_axis
        self.logger.debug(f"[_create_supplement_all_sides] dx={dx}, dy={dy}")
        return _create_supplement_all_sides(M, dx, dy, shade_gray_color)


def random_shift(
    self,
    M: np.ndarray,
    h: int,
    w: int,
    fill: int,
    is_right: Optional[bool] = None,
    **kwargs,
) -> np.ndarray:
    if is_right is True:
        dx, dy = random.randint(1, 4), random.randint(-4, 4)
    elif is_right is False:
        dx, dy = random.randint(-4, -1), random.randint(-4, 4)
    else:
        dx, dy = random.randint(-4, 4), random.randint(-4, 4)

    sx, sy = random.randint(0, abs(dx) * 2), random.randint(0, abs(dy) * 2)

    shifted = random_shift_engine(
        M=M,
        h=h,
        w=w,
        dx=dx,
        dy=dy,
        sx=sx,
        sy=sy,
        fill=fill,
    )

    if shifted.shape[:2] != (h, w):
        self.logger.warning(f"Shape mismatch: {shifted.shape} vs {(h, w)}")

    return shifted


@dataclass
@class_autologger
class RandomUniformProvider:
    low: float = 0.0
    high: float = 1.0

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.low > self.high:
            self.logger.warning(
                f"Logic error: low={self.low} > high={self.high}. Swapping values."
            )
            self.low, self.high = self.high, self.low

        self.logger.info(f"Validated provider with range [{self.low}, {self.high}].")

    def get_value(self) -> float:
        value = get_uniform_value(self.low, self.high)
        self.logger.debug(f"Generated value={value:.4f}")
        return value


@class_autologger
class NoiseAugmentation:
    std_provider: Optional[ParameterProviderProtocol] = None
    prob_provider: Optional[ParameterProviderProtocol] = None

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.std_provider is None:
            self.std_provider = RandomUniformProvider(0.1, 5.0)
        if self.prob_provider is None:
            self.prob_provider = RandomUniformProvider(0.01, 0.05)
        self.logger.info("Validated NoiseAugmentation providers.")

    def gaussian_noise(
        self, M: np.ndarray, std: Optional[float] = None, **kwargs
    ) -> np.ndarray:
        assert self.std_provider is not None
        actual_std = (
            float(std) if std is not None else float(self.std_provider.get_value())
        )

        self.logger.debug(f"[gaussian_noise] Applying std={actual_std:.4f}")

        noise_map = np.random.standard_normal(M.shape)
        return gaussian_noise(M, actual_std, noise_map)

    def salt_and_pepper(
        self, M: np.ndarray, prob: Optional[float] = None, **kwargs
    ) -> np.ndarray:
        assert self.prob_provider is not None
        actual_prob = (
            float(prob) if prob is not None else float(self.prob_provider.get_value())
        )

        self.logger.debug(f"[salt_and_pepper] Applying prob={actual_prob:.4f}")

        random_map = np.random.random(M.shape)
        return salt_and_pepper(M, actual_prob, random_map)


@class_autologger
@apply_to_methods(auto_fill_color, ["erode", "get_boundaries", "morphology_filter"])
@apply_to_methods(kernel_data_processing, ["dilate", "erode", "get_boundaries"])
class MorphologyAugmentation:
    logger: logging.Logger

    def _sliding_window_engine(
        self,
        M: np.ndarray,
        kernel_size: int,
        op_func: Optional[Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]] = None,
        pad_value: int = 0,
    ) -> np.ndarray:
        M_arr = np.asanyarray(M, dtype=np.uint8)
        h, w = M_arr.shape[:2]

        k_size = int(kernel_size)
        pad_before = (k_size - 1) // 2
        pad_after = k_size // 2

        actual_op = (
            op_func if op_func is not None else lambda win, axes: np.max(win, axis=axes)
        )
        op_name = getattr(actual_op, "__name__", "lambda")

        self.logger.debug(
            f"[_sliding_window_engine] Executing: kernel={k_size}, op={op_name}, pad_val={pad_value}"
        )

        result = sliding_window_engine(
            M_arr=M_arr,
            kernel_size=k_size,
            pad_before=pad_before,
            pad_after=pad_after,
            pad_value=int(pad_value),
            op_func=actual_op,
        )

        if result.shape[:2] != (h, w):
            self.logger.warning(f"Shape mismatch: {result.shape} vs {(h, w)}")

        self.logger.info(f"Engine processed {h}x{w} matrix.")
        return result

    def dilate(self, M: np.ndarray, kernel_size: int, **kwargs) -> np.ndarray:
        self.logger.debug(f"[dilate] kernel_size={kernel_size}")
        return dilate(M, int(kernel_size), self._sliding_window_engine)

    def erode(
        self, M: np.ndarray, kernel_size: int, fill: int = 0, **kwargs
    ) -> np.ndarray:
        self.logger.debug(f"[erode] kernel_size={kernel_size}, fill={fill}")
        return erode(M, int(kernel_size), int(fill), self._sliding_window_engine)

    def get_boundaries(
        self, M: np.ndarray, kernel_size: int = 2, **kwargs
    ) -> np.ndarray:
        self.logger.debug(f"[get_boundaries] kernel_size={kernel_size}")
        m_arr = np.asanyarray(M)
        eroded = self.erode(m_arr, kernel_size=int(kernel_size), **kwargs)
        return get_boundaries(m_arr, eroded)

    def morphology_filter(
        self,
        M: np.ndarray,
        kernel_size: int,
        fill: int = 0,
        mode: Literal["open", "close"] = "open",
        **kwargs,
    ) -> np.ndarray:
        self.logger.debug(f"[morphology_filter] mode={mode}, kernel={kernel_size}")
        target_mode = mode if mode in ["open", "close"] else "close"
        if mode not in ["open", "close"]:
            self.logger.warning(f"Unexpected mode '{mode}', falling back to 'close'")

        return morphology_filter(
            M=M,
            kernel_size=int(kernel_size),
            fill=int(fill),
            mode=target_mode,
            dilate_func=self.dilate,
            erode_func=self.erode,
        )
