import random
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
)

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from common_utils import class_autologger
from MyTorch import T2D, ImageGray, JsonData, Padded

from .base import AugmentationABC, GeometryABC, MorphologyABC, NoiseABC, ProviderABC
from .decorators import (
    apply_to_methods,
    auto_fill_color,
    kernel_data_processing,
    with_dimensions,
)
from .protocols import (
    GeometryAugmentationProtocol,
    MorphologyAugmentationProtocol,
    NoiseAugmentationProtocol,
    ParameterProviderProtocol,
)


def _pick_random_pipeline(
    available_keys: List[str],
    groups: Dict[str, List[Tuple[Callable, str]]],
    num_samples: int = 3,
) -> Tuple[List[Tuple[Callable, str]], List[str]]:
    selected_keys: List[str] = random.sample(
        available_keys, min(len(available_keys), num_samples)
    )
    pipeline: List[Tuple[Callable, str]] = [
        random.choice(groups[k]) for k in selected_keys
    ]
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
    m_arr: ImageGray,
    repeats: int,
    max_attempts: int,
    pipeline_provider: Callable[[], List[Tuple[Callable, str]]],
    orig_px: int,
    min_area_ratio: float = 0.5,
) -> Tuple[List[ImageGray], List[List[str]], int]:
    results: List[ImageGray] = list()
    histories: List[List[str]] = list()
    attempts: int = 0

    while len(results) < repeats and attempts < max_attempts:
        attempts += 1
        pipeline: List[Tuple[Callable, str]] = pipeline_provider()
        current_m: ImageGray = m_arr.copy()
        current_history: List[str] = [name for _, name in pipeline]

        for func, _ in pipeline:
            current_m = func(current_m)

        current_px: int = int(np.sum(current_m > 0))
        if current_px >= (orig_px * min_area_ratio):
            results.append(current_m)
            histories.append(current_history)

    return results, histories, attempts


def _get_trace_report(idx: int, history: List[str]) -> str:
    steps: str = " -> ".join([f"[{i + 1}] {name}" for i, name in enumerate(history)])
    return f"\n[PLOT {idx + 1}] Operation Trace:\n  {steps}"


def horizontal_flip(M: ImageGray) -> ImageGray:
    return np.asanyarray(M)[:, ::-1]


def vertical_flip(M: ImageGray) -> ImageGray:
    return np.asanyarray(M)[::-1, :]


def rotate_90(M: ImageGray, is_right: bool = True) -> ImageGray:
    k: int = -1 if is_right else 1
    return np.rot90(np.asanyarray(M), k=k).copy()


def _create_supplement_all_sides(
    M: ImageGray, dx: int, dy: int, shade_gray_color: int
) -> ImageGray:
    return np.pad(
        M,
        pad_width=((abs(dy), abs(dy)), (abs(dx), abs(dx))),
        mode="constant",
        constant_values=shade_gray_color,
    ).astype(np.float32)


def rotate_small_angle(
    M: ImageGray,
    h: int,
    w: int,
    cos_a: float,
    sin_a: float,
    cx: float,
    cy: float,
    new_m: ImageGray,
) -> ImageGray:
    y_new, x_new = np.indices((h, w))
    tx, ty = x_new - cx, y_new - cy
    xf = tx * cos_a + ty * sin_a + cx
    yf = -tx * sin_a + ty * cos_a + cy

    mask = (xf >= 0) & (xf < w - 1) & (yf >= 0) & (yf < h - 1)
    xi, yi = xf[mask].astype(int), yf[mask].astype(int)

    new_m[y_new[mask], x_new[mask]] = M[yi, xi]
    return new_m.astype(np.float32)


def random_shift_engine(
    M: ImageGray, h: int, w: int, dx: int, dy: int, sx: int, sy: int, fill: int
) -> ImageGray:
    padded = np.pad(
        M,
        pad_width=((abs(dy), abs(dy)), (abs(dx), abs(dx))),
        mode="constant",
        constant_values=fill,
    ).astype(np.float32)

    shifted: ImageGray = padded[sy : sy + h, sx : sx + w].copy()
    return shifted


def get_uniform_value(low: float, high: float) -> float:
    return random.uniform(low, high)


def gaussian_noise(M: ImageGray, std: float, noise_map: ImageGray) -> ImageGray:
    return np.clip(M + (noise_map * std), 0.0, 255.0).astype(np.float32)


def salt_and_pepper(M: ImageGray, prob: float, random_map: ImageGray) -> ImageGray:
    result: ImageGray = M.copy()
    salt_threshold: float = 1.0 - prob / 2.0
    pepper_threshold: float = prob / 2.0
    result[random_map < pepper_threshold] = 0.0
    result[random_map > salt_threshold] = 255.0
    return result


def dilate(M: ImageGray, kernel_size: int, sliding_window_func: Callable) -> ImageGray:
    return sliding_window_func(M, kernel_size, pad_value=0.0)


def erode(
    M: ImageGray, kernel_size: int, fill: int, sliding_window_func: Callable
) -> ImageGray:
    return sliding_window_func(
        M,
        kernel_size,
        op_func=lambda win, axes: np.min(win, axis=axes),
        pad_value=float(fill),
    )


def get_boundaries(M: ImageGray, eroded: ImageGray) -> ImageGray:
    diff = M.astype(np.float32) - eroded.astype(np.float32)
    return np.clip(diff, 0.0, 255.0).astype(np.float32)


def morphology_filter(
    M: ImageGray,
    kernel_size: int,
    fill: int,
    mode: str,
    dilate_func: Callable,
    erode_func: Callable,
) -> ImageGray:
    if mode == "open":
        return dilate_func(erode_func(M, kernel_size, fill), kernel_size)
    return erode_func(dilate_func(M, kernel_size), kernel_size, fill)


def sliding_window_engine(
    M_arr: ImageGray,
    kernel_size: int,
    pad_before: int,
    pad_after: int,
    pad_value: int,
    op_func: Callable,
) -> ImageGray:
    h, w = M_arr.shape
    padded: Padded = np.pad(
        M_arr,
        pad_width=((pad_before, pad_after), (pad_before, pad_after)),
        mode="constant",
        constant_values=float(pad_value),
    )
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    result: ImageGray = op_func(windows, (2, 3)).astype(np.float32)
    return result[:h, :w]


@class_autologger
@apply_to_methods(
    [auto_fill_color, with_dimensions], ["rotate_small_angle", "random_shift"]
)
@dataclass(frozen=True, slots=True)
class GeometryAugmentation(GeometryABC):
    angle_provider: ParameterProviderProtocol = field(
        default_factory=lambda: RandomUniformProvider(-5.0, 5.0)
    )
    shift_provider: ParameterProviderProtocol = field(
        default_factory=lambda: RandomUniformProvider(-10.0, 10.0)
    )
    _transformation_matrix_cache: T2D = field(
        init=False, repr=False, default_factory=lambda: np.array([], dtype=np.float32)
    )

    def __post_init__(self) -> None:
        super().__post_init__()

    def horizontal_flip(self, M: ImageGray) -> ImageGray:
        return horizontal_flip(M)

    def vertical_flip(self, M: ImageGray) -> ImageGray:
        return vertical_flip(M)

    def rotate_90(self, M: ImageGray, is_right: bool = True) -> ImageGray:
        return rotate_90(M, is_right)

    def rotate_small_angle(
        self,
        M: ImageGray,
        h: int,
        w: int,
        params: JsonData,
        angle: float = 0.0,
        fill: int = 0,
    ) -> ImageGray:
        assert isinstance(params, dict)

        return rotate_small_angle(
            M=M,
            h=h,
            w=w,
            cos_a=float(params["cos_a"]),
            sin_a=float(params["sin_a"]),
            cx=float(params["cx"]),
            cy=float(params["cy"]),
            new_m=params["new_matrix"],
        )

    def _create_supplement_all_sides(
        self, M: ImageGray, x_y_axis: Tuple[int, int], shade_gray_color: int
    ) -> ImageGray:
        dx, dy = x_y_axis
        return _create_supplement_all_sides(M, dx, dy, shade_gray_color)

    def random_shift(
        self,
        M: ImageGray,
        h: int,
        w: int,
        fill: int = 0,
        is_right: bool = True,
    ) -> ImageGray:
        dx, dy = (
            (random.randint(1, 4), random.randint(-4, 4))
            if is_right
            else (random.randint(-4, -1), random.randint(-4, 4))
        )
        sx, sy = random.randint(0, abs(dx) * 2), random.randint(0, abs(dy) * 2)
        return random_shift_engine(M=M, h=h, w=w, dx=dx, dy=dy, sx=sx, sy=sy, fill=fill)


@dataclass(frozen=True, slots=True)
class RandomUniformProvider(ProviderABC):
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.low > self.high:
            object.__setattr__(self, "low", self.high)
            object.__setattr__(self, "high", self.low)

    def get_value(self) -> float:
        return get_uniform_value(self.low, self.high)


@dataclass(frozen=True, slots=True)
class NoiseAugmentation(NoiseABC):
    std_provider: ParameterProviderProtocol = field(
        default_factory=lambda: RandomUniformProvider(0.1, 5.0)
    )
    prob_provider: ParameterProviderProtocol = field(
        default_factory=lambda: RandomUniformProvider(0.01, 0.05)
    )

    def __post_init__(self) -> None:
        super().__post_init__()

    def gaussian_noise(self, M: ImageGray, std: float = 0.0) -> ImageGray:
        actual_std: float = std if std > 0.0 else self.std_provider.get_value()
        noise_map: ImageGray = np.random.standard_normal(M.shape).astype(np.float32)
        return gaussian_noise(M, actual_std, noise_map)

    def salt_and_pepper(self, M: ImageGray, prob: float = 0.0) -> ImageGray:
        actual_prob: float = prob if prob > 0.0 else self.prob_provider.get_value()
        random_map: ImageGray = np.random.random(M.shape).astype(np.float32)
        return salt_and_pepper(M, actual_prob, random_map)


@class_autologger
@apply_to_methods(auto_fill_color, ["erode", "get_boundaries", "morphology_filter"])
@apply_to_methods(kernel_data_processing, ["dilate", "erode", "get_boundaries"])
@dataclass(frozen=True, slots=True)
class MorphologyAugmentation(MorphologyABC):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _sliding_window_engine(
        self, M: ImageGray, kernel_size: int, op_func: Callable, pad_value: int = 0
    ) -> ImageGray:
        pad_before: int = (kernel_size - 1) // 2
        pad_after: int = kernel_size // 2
        return sliding_window_engine(
            M_arr=M,
            kernel_size=kernel_size,
            pad_before=pad_before,
            pad_after=pad_after,
            pad_value=pad_value,
            op_func=op_func,
        )

    def dilate(self, M: ImageGray, kernel_size: int) -> ImageGray:
        return dilate(M, kernel_size, self._sliding_window_engine)

    def erode(self, M: ImageGray, kernel_size: int, fill: int = 0) -> ImageGray:
        return erode(M, kernel_size, fill, self._sliding_window_engine)

    def get_boundaries(self, M: ImageGray, kernel_size: int = 2) -> ImageGray:
        eroded: ImageGray = self.erode(M, kernel_size=kernel_size)
        return get_boundaries(M, eroded)

    def morphology_filter(
        self, M: ImageGray, kernel_size: int, fill: int = 0, mode: str = "open"
    ) -> ImageGray:
        return morphology_filter(
            M=M,
            kernel_size=kernel_size,
            fill=fill,
            mode=mode,
            dilate_func=self.dilate,
            erode_func=self.erode,
        )


@dataclass(frozen=True, slots=True)
class DataAugmentation(AugmentationABC):
    geometry: GeometryAugmentationProtocol = field(default_factory=GeometryAugmentation)
    noise: NoiseAugmentationProtocol = field(default_factory=NoiseAugmentation)
    morphology: MorphologyAugmentationProtocol = field(
        default_factory=MorphologyAugmentation
    )
    _cached_groups: Dict[str, List[Tuple[Callable, str]]] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        groups: Dict[str, List[Tuple[Callable, str]]] = _get_augmentation_groups(
            self.geometry, self.noise, self.morphology
        )
        object.__setattr__(self, "_cached_groups", groups)

    def augment(
        self, M: ImageGray, repeats: int = 1, debug: bool = False
    ) -> List[ImageGray]:
        orig_px: int = int(np.sum(M > 0))
        max_attempts: int = repeats * 100

        results, histories, attempts = augment_engine(
            m_arr=M,
            repeats=repeats,
            max_attempts=max_attempts,
            pipeline_provider=self._pick_random_pipeline,
            orig_px=orig_px,
        )

        if debug:
            self._display_debug_plots(results, histories)

        return results

    def _pick_random_pipeline(self) -> List[Tuple[Callable, str]]:
        available_keys: List[str] = list(self._cached_groups.keys())
        pipeline, _ = _pick_random_pipeline(available_keys, self._cached_groups)
        return pipeline

    def _display_debug_plots(
        self, result: List[ImageGray], histories: List[List[str]]
    ) -> None:
        for idx, history in enumerate(histories):
            print(_get_trace_report(idx, history))
