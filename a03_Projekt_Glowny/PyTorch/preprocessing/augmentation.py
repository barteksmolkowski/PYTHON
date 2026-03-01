import logging
import random
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


@class_autologger
class DataAugmentation:
    logger: logging.Logger

    def __init__(
        self,
        geometry: Optional[GeometryAugmentationProtocol] = None,
        noise: Optional[NoiseAugmentationProtocol] = None,
        morphology: Optional[MorphologyAugmentationProtocol] = None,
    ):
        if geometry is None:
            self.logger.debug(
                "[__init__] geometry is None, using default GeometryAugmentation"
            )
        self.geometry = geometry or GeometryAugmentation()

        if noise is None:
            self.logger.debug(
                "[__init__] noise is None, using default NoiseAugmentation"
            )
        self.noise = noise or NoiseAugmentation()

        if morphology is None:
            self.logger.debug(
                "[__init__] morphology is None, using default MorphologyAugmentation"
            )
        self.morphology = morphology or MorphologyAugmentation()

        self._cached_groups = self._get_augmentation_groups()
        self.logger.info(
            f"[__init__] Validated initialization with {len(self._cached_groups)} groups."
        )

    def _get_augmentation_groups(self) -> Dict[str, List[Tuple[Callable, str]]]:
        self.logger.debug(
            f"[_get_augmentation_groups] Building groups with: "
            f"geometry={type(self.geometry).__name__}, "
            f"noise={type(self.noise).__name__}, "
            f"morphology={type(self.morphology).__name__}"
        )

        groups = {
            "Flip": [
                (
                    lambda m, **kwargs: self.geometry.horizontal_flip(m, **kwargs),
                    "H-Flip",
                ),
                (
                    lambda m, **kwargs: self.geometry.vertical_flip(m, **kwargs),
                    "V-Flip",
                ),
            ],
            "Rotation": [
                (
                    lambda m, **kwargs: self.geometry.rotate_90(
                        m, is_right=True, **kwargs
                    ),
                    "Rot90-R",
                ),
                (
                    lambda m, **kwargs: self.geometry.rotate_small_angle(
                        m, **kwargs, is_right=True
                    ),
                    "RotSmall-R",
                ),
            ],
            "Morphology": [
                (
                    lambda m, **kwargs: self.morphology.dilate(
                        m, kernel_size=1, **kwargs
                    ),
                    "Dilate",
                ),
                (
                    lambda m, **kwargs: self.morphology.morphology_filter(
                        m, 1, mode="close", **kwargs
                    ),
                    "Closing",
                ),
            ],
            "Noise_Shift": [
                (
                    lambda m, **kwargs: self.noise.salt_and_pepper(m, **kwargs),
                    "S&P-Noise",
                ),
                (
                    lambda m, **kwargs: self.geometry.random_shift(
                        m, is_right=True, **kwargs
                    ),
                    "Shift-R",
                ),
            ],
        }

        self.logger.info(
            f"[_get_augmentation_groups] Built and silenced {len(groups)} groups."
        )
        return groups

    def _pick_random_pipeline(self) -> List[Tuple[Callable, str]]:
        available_keys = list(self._cached_groups.keys())

        selected_keys = random.sample(available_keys, 3)
        self.logger.debug(
            f"[_pick_random_pipeline] Selected keys: {selected_keys} from available: {available_keys}"
        )

        pipeline = []
        for k in selected_keys:
            choice = random.choice(self._cached_groups[k])
            self.logger.debug(
                f"[_pick_random_pipeline] Picked transformation: '{choice[1]}' from group: '{k}'"
            )
            pipeline.append(choice)

        return pipeline

    @get_number_repeats
    def augment(
        self, M: Mtx, repeats: Optional[int] = None, debug: bool = False
    ) -> MtxList:
        if repeats is None:
            self.logger.debug(
                "[augment] repeats is None, setting default value repeats=1"
            )
            repeats = 1

        result: MtxList = []
        histories: list[list[str]] = []

        M_arr = np.asanyarray(M).astype(np.uint8)
        orig_px = np.sum(M_arr > 0)

        attempts = 0
        max_attempts = repeats * 100
        self.logger.debug(
            f"[augment] Starting augmentation loop: repeats={repeats}, max_attempts={max_attempts}, orig_px={orig_px}"
        )

        while len(result) < repeats and attempts < max_attempts:
            attempts += 1
            pipeline = self._pick_random_pipeline()
            names = [name for _, name in pipeline]

            if names.count("Rotation") > 1:
                self.logger.debug(
                    f"[augment] Skipping pipeline {names}: too many Rotation groups"
                )
                continue

            rot_names = [n for n in names if "Rot" in n]
            if len(rot_names) > 1:
                self.logger.debug(
                    f"[augment] Skipping pipeline {names}: multiple rotation transforms detected {rot_names}"
                )
                continue

            if "Boundaries" in names and "Erode" in names:
                self.logger.debug(
                    f"[augment] Skipping pipeline {names}: incompatible Boundaries + Erode"
                )
                continue
            if "Dilate" in names and "Erode" in names:
                self.logger.debug(
                    f"[augment] Skipping pipeline {names}: incompatible Dilate + Erode"
                )
                continue
            if "V-Flip" in names:
                self.logger.debug(
                    f"[augment] Skipping pipeline {names}: V-Flip is blacklisted"
                )
                continue
            if any("Rot" in n for n in names) and any(
                m in names for m in ["Dilate", "Erode", "Opening"]
            ):
                self.logger.debug(
                    f"[augment] Skipping pipeline {names}: Rotation cannot be combined with morphology"
                )
                continue

            new_m = M_arr.copy()
            for func, _ in pipeline:
                new_m = func(new_m)

            final_px = np.sum(new_m > 0)
            retention = final_px / orig_px if orig_px > 0 else 0

            if not (0.2 < retention < 3.0):
                self.logger.debug(
                    f"[augment] Rejected by retention: {retention:.2f} (orig_px={orig_px}, final_px={final_px})"
                )
                continue

            coords = np.argwhere(new_m > 0)
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                h, w = (y_max - y_min + 1), (x_max - x_min + 1)

                aspect_ratio = h / (w + 1e-8)

                if aspect_ratio < 0.2:
                    self.logger.debug(
                        f"[augment] Rejected by aspect_ratio: {aspect_ratio:.2f} for pipeline {names}"
                    )
                    continue

                result.append(new_m.astype(np.uint8))
                self.logger.debug(
                    f"[augment] Successfully added sample {len(result)}/{repeats} using {names}"
                )
                if debug:
                    histories.append(names)
            else:
                self.logger.debug(
                    f"[augment] Rejected: Resulting matrix is empty after pipeline {names}"
                )

        if attempts >= max_attempts:
            self.logger.warning(
                f"[augment] Max attempts reached: {attempts}/{max_attempts}. Collected only {len(result)}/{repeats} samples."
            )

        if debug:
            self.logger.info(f"[augment] Entering debug mode: items={len(result)}")
            self._display_debug_plots(result, histories)

        self.logger.info(f"[augment] Validated {len(result)} augmented items.")
        return result

    def _display_debug_plots(self, result: list, histories: list):
        cols = 10
        rows = (len(result) + cols - 1) // cols

        self.logger.debug(
            f"[_display_debug_plots] Preparing grid with rows={rows}, cols={cols} for items={len(result)}"
        )

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(15, 1.6 * rows),
            gridspec_kw={"wspace": 0.1, "hspace": 0.4},
            squeeze=False,
        )

        axes_flat = axes.flatten()

        for i in range(len(result)):
            ax: Axes = axes_flat[i]
            ax.imshow(result[i], cmap="gray")
            ax.set_title(f"NR: {i + 1}", fontsize=9, fontweight="bold")
            ax.axis("off")

        for j in range(len(result), len(axes_flat)):
            extra_ax: Axes = axes_flat[j]
            extra_ax.axis("off")

        plt.show()
        self.logger.info(
            f"[_display_debug_plots] Rendered debug grid for {len(result)} items."
        )

        print("\n" + "=" * 50 + " INTERACTIVE DEBUG MODE (2026) " + "=" * 50)

        while True:
            user_input = input(
                "\nEnter plot indices that look incorrect (e.g., 1, 5, 12) or 'q' to quit: "
            )

            if user_input.lower() == "q":
                self.logger.debug(
                    f"[_display_debug_plots] Loop termination triggered by input='{user_input}'"
                )
                print("Exiting Debug Mode...")
                break

            try:
                selected_indices = [
                    int(x.strip()) - 1 for x in user_input.split(",") if x.strip()
                ]
                self.logger.debug(
                    f"[_display_debug_plots] Parsed selected_indices={[i + 1 for i in selected_indices]}"
                )

                for idx in selected_indices:
                    if 0 <= idx < len(histories):
                        print(f"\n[PLOT {idx + 1}] Operation Trace:")
                        steps = histories[idx]
                        formatted_history = " -> ".join(
                            [f"[{i + 1}] {name}" for i, name in enumerate(steps)]
                        )
                        print(f"  {formatted_history}")
                        self.logger.debug(
                            f"[_display_debug_plots] Trace displayed for index={idx + 1}, steps={steps}"
                        )
                    else:
                        print(f"[!] Index {idx + 1} is out of range.")
                        self.logger.warning(
                            f"[_display_debug_plots] Out of bounds: index={idx + 1} while histories_len={len(histories)}"
                        )

                self.logger.info(
                    f"[_display_debug_plots] Successfully processed {len(selected_indices)} trace requests."
                )

            except ValueError:
                print("[!] Error! Please enter comma-separated numbers or 'q'.")
                self.logger.error(
                    f"[_display_debug_plots] Invalid input format: input='{user_input}'"
                )


@class_autologger
@apply_to_methods(
    [auto_fill_color, with_dimensions], ["rotate_small_angle", "random_shift"]
)
class GeometryAugmentation:
    logger: logging.Logger

    def horizontal_flip(self, M: np.ndarray) -> np.ndarray:
        self.logger.debug(f"[horizontal_flip] Flipping matrix with shape: {M.shape}")
        return np.asanyarray(M)[:, ::-1]

    def vertical_flip(self, M: np.ndarray) -> np.ndarray:
        self.logger.debug(
            f"[vertical_flip] Flipping matrix vertically with shape: {M.shape}"
        )
        return np.asanyarray(M)[::-1, :]

    def rotate_90(self, M: np.ndarray, is_right: bool = True) -> np.ndarray:
        M = np.asanyarray(M)

        if is_right:
            self.logger.debug(f"[rotate_90] Direction: clockwise, is_right={is_right}")
            k = -1
        else:
            self.logger.debug(
                f"[rotate_90] Direction: counter-clockwise, is_right={is_right}"
            )
            k = 1

        return np.rot90(M, k=k).copy()

    @prepare_angle
    @prepare_values
    def rotate_small_angle(
        self,
        M: np.ndarray,
        h: int,
        w: int,
        params: dict,
        angle: float = 0.0,
        fill: int = 0,
        **kwargs,
    ) -> np.ndarray:
        c, s = params["cos_a"], params["sin_a"]
        cx, cy = params["cx"], params["cy"]
        new_m: np.ndarray = params["new_matrix"]

        self.logger.debug(
            f"[rotate_small_angle] Rotating {h}x{w} matrix. "
            f"angle={angle}, fill={fill}, center=({cx}, {cy})"
        )

        y_new, x_new = np.indices((h, w))
        tx, ty = x_new - cx, y_new - cy
        xf = tx * c + ty * s + cx
        yf = -tx * s + ty * c + cy

        mask = (xf >= 0) & (xf < w - 1) & (yf >= 0) & (yf < h - 1)

        xi, yi = xf[mask].astype(int), yf[mask].astype(int)
        new_m[y_new[mask], x_new[mask]] = M[yi, xi]

        self.logger.info(f"[rotate_small_angle] Validated rotation for {h}x{w} matrix.")
        return new_m.astype(np.uint8)

    def _create_supplement_all_sides(
        self, M: np.ndarray, x_y_axis: Tuple[int, int], shade_gray_color: int
    ) -> np.ndarray:
        dx, dy = x_y_axis

        self.logger.debug(
            f"[_create_supplement_all_sides] Padding matrix (shape={M.shape}) "
            f"with dx={abs(dx)}, dy={abs(dy)}, color={shade_gray_color}"
        )

        return np.pad(
            M,
            pad_width=((abs(dy), abs(dy)), (abs(dx), abs(dx))),
            mode="constant",
            constant_values=shade_gray_color,
        ).astype(np.uint8)

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
            self.logger.debug(
                f"[random_shift] Direction fixed to RIGHT: is_right={is_right}, sampled dx={dx}, dy={dy}"
            )
        elif is_right is False:
            dx, dy = random.randint(-4, -1), random.randint(-4, 4)
            self.logger.debug(
                f"[random_shift] Direction fixed to LEFT: is_right={is_right}, sampled dx={dx}, dy={dy}"
            )
        else:
            dx, dy = random.randint(-4, 4), random.randint(-4, 4)
            self.logger.debug(
                f"[random_shift] Direction is RANDOM: is_right={is_right}, sampled dx={dx}, dy={dy}"
            )

        padded = self._create_supplement_all_sides(M, (dx, dy), fill)

        sx, sy = random.randint(0, abs(dx) * 2), random.randint(0, abs(dy) * 2)
        self.logger.debug(
            f"[random_shift] Cropping from padded matrix: start_x={sx}, start_y={sy} for target_size={w}x{h}"
        )

        shifted = padded[sy : sy + h, sx : sx + w].copy()

        if shifted.shape != (h, w):
            self.logger.warning(
                f"[random_shift] Output shape mismatch: expected {(h, w)}, got {shifted.shape}"
            )
        else:
            self.logger.info(
                f"[random_shift] Validated shifted matrix of size {w}x{h}."
            )

        return shifted


@class_autologger
class RandomUniformProvider:
    logger: logging.Logger

    def __init__(self, low: float = 0.0, high: float = 1.0):
        if low > high:
            self.logger.warning(
                f"[__init__] Logic error: low={low} is greater than high={high}. Swapping values."
            )

        self.low = low
        self.high = high
        self.logger.info(
            f"[__init__] Validated provider with range [{self.low}, {self.high}]."
        )

    def get_value(self) -> float:
        value = random.uniform(self.low, self.high)
        self.logger.debug(
            f"[get_value] Generated value={value:.4f} for range=[{self.low}, {self.high}]"
        )
        return value


@class_autologger
class NoiseAugmentation:
    logger: logging.Logger

    def __init__(
        self,
        std_provider: Optional[ParameterProviderProtocol] = None,
        prob_provider: Optional[ParameterProviderProtocol] = None,
    ):
        if std_provider is None:
            self.logger.debug(
                "[__init__] std_provider is None, using default RandomUniformProvider(0.1, 5.0)"
            )
        self.std_provider = std_provider or RandomUniformProvider(0.1, 5.0)

        if prob_provider is None:
            self.logger.debug(
                "[__init__] prob_provider is None, using default RandomUniformProvider(0.01, 0.05)"
            )
        self.prob_provider = prob_provider or RandomUniformProvider(0.01, 0.05)

        self.logger.info("[__init__] Validated NoiseAugmentation providers.")

    def gaussian_noise(
        self, M: np.ndarray, std: Optional[float] = None, **kwargs
    ) -> np.ndarray:
        if std is not None:
            actual_std = std
            self.logger.debug(f"[gaussian_noise] Using explicit std={actual_std}")
        else:
            actual_std = self.std_provider.get_value()
            self.logger.debug(
                f"[gaussian_noise] std not provided, fetched actual_std={actual_std:.4f}"
            )

        np_M = np.asanyarray(M, dtype=np.float32)
        noise = np.random.normal(0, actual_std, np_M.shape)

        self.logger.info(
            f"[gaussian_noise] Validated Gaussian noise application for shape={np_M.shape} with std={actual_std:.4f}"
        )
        return np.clip(np_M + noise, 0, 255).astype(np.uint8)

    def salt_and_pepper(
        self, M: np.ndarray, prob: Optional[float] = None, **kwargs
    ) -> np.ndarray:
        if prob is not None:
            actual_prob = prob
            self.logger.debug(f"[salt_and_pepper] Using explicit prob={actual_prob}")
        else:
            actual_prob = self.prob_provider.get_value()
            self.logger.debug(
                f"[salt_and_pepper] prob not provided, fetched actual_prob={actual_prob:.4f}"
            )

        result = np.copy(M).astype(np.uint8)
        random_map = np.random.random(result.shape)

        salt_threshold = 1 - actual_prob / 2
        pepper_threshold = actual_prob / 2

        self.logger.debug(
            f"[salt_and_pepper] Applying thresholds: pepper < {pepper_threshold:.4f}, salt > {salt_threshold:.4f} for shape={result.shape}"
        )

        result[random_map < pepper_threshold] = 0
        result[random_map > salt_threshold] = 255

        self.logger.info(
            f"[salt_and_pepper] Validated S&P noise for matrix shape={result.shape}."
        )
        return result


@class_autologger
@apply_to_methods(auto_fill_color, ["erode", "get_boundaries", "morphology_filter"])
@apply_to_methods(kernel_data_processing, ["dilate", "erode", "get_boundaries"])
class MorphologyAugmentation:
    logger: logging.Logger

    def _sliding_window_engine(
        self,
        M: np.ndarray,
        kernel_size: int,
        op_func: Optional[Callable[[np.ndarray, Tuple[int, int]], np.ndarray]] = None,
        pad_value: int = 0,
    ) -> np.ndarray:
        M_arr = np.asanyarray(M, dtype=np.uint8)
        h, w = M_arr.shape[:2]

        pad_before = (kernel_size - 1) // 2
        pad_after = kernel_size // 2

        self.logger.debug(
            f"[_sliding_window_engine] Padding {h}x{w} matrix: kernel_size={kernel_size}, "
            f"pad_before={pad_before}, pad_after={pad_after}, pad_value={pad_value}"
        )

        padded = np.pad(
            M_arr,
            pad_width=((pad_before, pad_after), (pad_before, pad_after)),
            mode="constant",
            constant_values=pad_value,
        )

        windows = sliding_window_view(padded, (kernel_size, kernel_size))

        if op_func is None:
            self.logger.debug(
                "[_sliding_window_engine] No op_func provided, defaulting to np.max (Dilation logic)"
            )
            op_func = lambda win, axes: np.max(win, axis=axes)
        else:
            op_name = op_func.__name__ if hasattr(op_func, "__name__") else "lambda"
            self.logger.debug(
                f"[_sliding_window_engine] Using custom operator: op_func='{op_name}'"
            )

        result = op_func(windows, (2, 3)).astype(np.uint8)

        if result.shape[:2] != (h, w):
            self.logger.warning(
                f"[_sliding_window_engine] Shape mismatch: result={result.shape} vs input={(h, w)}. Applying crop."
            )

        self.logger.info(
            f"[_sliding_window_engine] Validated engine execution for {h}x{w} matrix."
        )
        return result[:h, :w]

    def dilate(self, M: np.ndarray, kernel_size: int, **kwargs) -> np.ndarray:
        self.logger.debug(f"[dilate] Starting dilation: kernel_size={kernel_size}")
        result = self._sliding_window_engine(M, kernel_size, pad_value=0)

        self.logger.info(f"[dilate] Validated dilation for matrix shape={M.shape}.")
        return result

    def erode(
        self, M: np.ndarray, kernel_size: int, fill: int = 0, **kwargs
    ) -> np.ndarray:
        self.logger.debug(
            f"[erode] Executing erosion: kernel_size={kernel_size}, fill_color={fill}"
        )
        result = self._sliding_window_engine(
            M,
            kernel_size,
            op_func=lambda win, axes: np.min(win, axis=axes),
            pad_value=fill,
        )

        self.logger.info(f"[erode] Validated erosion for matrix shape={M.shape}.")
        return result

    def get_boundaries(
        self, M: np.ndarray, kernel_size: int = 2, **kwargs
    ) -> np.ndarray:
        M_arr = np.asanyarray(M)

        eroded = self.erode(M_arr, kernel_size=kernel_size, **kwargs)
        self.logger.debug(
            f"[get_boundaries] Erosion step completed: kernel_size={kernel_size}"
        )

        diff = M_arr.astype(np.int16) - eroded.astype(np.int16)
        boundary_pixels = np.sum(diff > 0)

        self.logger.debug(
            f"[get_boundaries] Difference calculated: boundary_pixels={boundary_pixels}"
        )

        result = np.clip(diff, 0, 255).astype(np.uint8)
        self.logger.info(f"[get_boundaries] Validated boundaries for shape={M.shape}.")
        return result

    def morphology_filter(
        self,
        M: np.ndarray,
        kernel_size: int,
        fill: int = 0,
        mode: Literal["open", "close"] = "open",
        **kwargs,
    ) -> np.ndarray:
        if mode == "open":
            self.logger.debug(
                f"[morphology_filter] Selected 'open' (erode -> dilate): kernel_size={kernel_size}, fill={fill}"
            )
            result = self.dilate(self.erode(M, kernel_size, fill=fill), kernel_size)
            self.logger.info(f"[morphology_filter] Validated 'open' filter.")
            return result

        if mode == "close":
            self.logger.debug(
                f"[morphology_filter] Selected 'close' (dilate -> erode): kernel_size={kernel_size}, fill={fill}"
            )
            result = self.erode(self.dilate(M, kernel_size), kernel_size, fill=fill)
            self.logger.info(f"[morphology_filter] Validated 'close' filter.")
            return result

        self.logger.warning(
            f"[morphology_filter] Unexpected mode='{mode}'. Fallback to 'close' logic."
        )
        return self.erode(self.dilate(M, kernel_size), kernel_size, fill=fill)
