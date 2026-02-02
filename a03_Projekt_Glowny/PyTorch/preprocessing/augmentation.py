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


class DataAugmentation:
    def __init__(
        self,
        geometry: Optional[GeometryAugmentationProtocol] = None,
        noise: Optional[NoiseAugmentationProtocol] = None,
        morphology: Optional[MorphologyAugmentationProtocol] = None,
    ):
        self.geometry = geometry or GeometryAugmentation()
        self.noise = noise or NoiseAugmentation()
        self.morphology = morphology or MorphologyAugmentation()

        self._cached_groups = self._get_augmentation_groups()

    def _get_augmentation_groups(self) -> Dict[str, List[Tuple[Callable, str]]]:
        return {
            "Flip": [
                (lambda m: self.geometry.horizontal_flip(m), "H-Flip"),
                (lambda m: self.geometry.vertical_flip(m), "V-Flip"),
            ],
            "Rotation": [
                (lambda m: self.geometry.rotate_90(m, is_right=True), "Rot90-R"),
                (
                    lambda m: self.geometry.rotate_small_angle(m, is_right=True),
                    "RotSmall-R",
                ),
            ],
            "Morphology": [
                (lambda m: self.morphology.dilate(m, kernel_size=1), "Dilate"),
                (
                    lambda m: self.morphology.morphology_filter(m, 1, mode="close"),
                    "Closing",
                ),
            ],
            "Noise_Shift": [
                (lambda m: self.noise.salt_and_pepper(m), "S&P-Noise"),
                (lambda m: self.geometry.random_shift(m, is_right=True), "Shift-R"),
            ],
        }

    def _pick_random_pipeline(self) -> List[Tuple[Callable, str]]:
        available_keys = list(self._cached_groups.keys())
        selected_keys = random.sample(available_keys, 3)
        return [random.choice(self._cached_groups[k]) for k in selected_keys]

    @get_number_repeats
    def augment(
        self, M: Mtx, repeats: Optional[int] = None, debug: bool = False
    ) -> MtxList:
        if repeats is None:
            repeats = 1

        result: MtxList = []
        histories: list[list[str]] = []

        M_arr = np.asanyarray(M).astype(np.uint8)
        orig_px = np.sum(M_arr > 0)

        attempts = 0
        max_attempts = repeats * 100

        while len(result) < repeats and attempts < max_attempts:
            attempts += 1
            pipeline = self._pick_random_pipeline()
            names = [name for _, name in pipeline]

            if names.count("Rotation") > 1:
                continue

            rot_names = [n for n in names if "Rot" in n]
            if len(rot_names) > 1:
                continue

            if "Boundaries" in names and "Erode" in names:
                continue
            if "Dilate" in names and "Erode" in names:
                continue
            if "V-Flip" in names:
                continue
            if any("Rot" in n for n in names) and any(
                m in names for m in ["Dilate", "Erode", "Opening"]
            ):
                continue

            new_m = M_arr.copy()
            for func, _ in pipeline:
                new_m = func(new_m)

            final_px = np.sum(new_m > 0)
            retention = final_px / orig_px if orig_px > 0 else 0

            if 0.2 < retention < 3.0:
                coords = np.argwhere(new_m > 0)

                if coords.size > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    h, w = (y_max - y_min + 1), (x_max - x_min + 1)

                    aspect_ratio = h / (w + 1e-8)

                    if aspect_ratio < 0.2:
                        continue

                    result.append(new_m.astype(np.uint8))
                    if debug:
                        histories.append(names)

        if debug:
            self._display_debug_plots(result, histories)

        return result

    def _display_debug_plots(self, result: list, histories: list):
        cols = 10
        rows = (len(result) + cols - 1) // cols

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
            ax.set_title(f"NR: {i+1}", fontsize=9, fontweight="bold")
            ax.axis("off")

        for j in range(len(result), len(axes_flat)):
            extra_ax: Axes = axes_flat[j]
            extra_ax.axis("off")

        plt.show()

        print("\n" + "=" * 50 + " INTERACTIVE DEBUG MODE (2026) " + "=" * 50)

        while True:
            user_input = input(
                "\nEnter plot indices that look incorrect (e.g., 1, 5, 12) or 'q' to quit: "
            )

            if user_input.lower() == "q":
                print("Exiting Debug Mode...")
                break

            try:
                selected_indices = [
                    int(x.strip()) - 1 for x in user_input.split(",") if x.strip()
                ]

                for idx in selected_indices:
                    if 0 <= idx < len(histories):
                        print(f"\n[PLOT {idx+1}] Operation Trace:")
                        steps = histories[idx]
                        formatted_history = " -> ".join(
                            [f"[{i+1}] {name}" for i, name in enumerate(steps)]
                        )
                        print(f"  {formatted_history}")
                    else:
                        print(f"[!] Index {idx+1} is out of range.")
            except ValueError:
                print("[!] Error! Please enter comma-separated numbers or 'q'.")


@apply_to_methods(
    [auto_fill_color, with_dimensions], ["rotate_small_angle", "random_shift"]
)
class GeometryAugmentation:
    def horizontal_flip(self, M: np.ndarray) -> np.ndarray:
        return np.asanyarray(M)[:, ::-1]

    def vertical_flip(self, M: np.ndarray) -> np.ndarray:
        return np.asanyarray(M)[::-1, :]

    def rotate_90(self, M: np.ndarray, is_right: bool = True) -> np.ndarray:
        M = np.asanyarray(M)
        k = -1 if is_right else 1
        return np.rot90(M, k=k).copy()

    @prepare_angle
    @prepare_values
    def rotate_small_angle(
        self, M: np.ndarray, h: int, w: int, params: dict, **kwargs
    ) -> np.ndarray:
        c, s = params["cos_a"], params["sin_a"]
        cx, cy = params["cx"], params["cy"]
        new_m: np.ndarray = params["new_matrix"]

        y_new, x_new = np.indices((h, w))
        tx, ty = x_new - cx, y_new - cy
        xf = tx * c + ty * s + cx
        yf = -tx * s + ty * c + cy

        mask = (xf >= 0) & (xf < w - 1) & (yf >= 0) & (yf < h - 1)
        xi, yi = xf[mask].astype(int), yf[mask].astype(int)
        new_m[y_new[mask], x_new[mask]] = M[yi, xi]

        return new_m.astype(np.uint8)

    def _create_supplement_all_sides(
        self, M: np.ndarray, x_y_axis: Tuple[int, int], shade_gray_color: int
    ) -> np.ndarray:
        dx, dy = x_y_axis
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
        elif is_right is False:
            dx, dy = random.randint(-4, -1), random.randint(-4, 4)
        else:
            dx, dy = random.randint(-4, 4), random.randint(-4, 4)

        padded = self._create_supplement_all_sides(M, (dx, dy), fill)

        sx, sy = random.randint(0, abs(dx) * 2), random.randint(0, abs(dy) * 2)
        return padded[sy : sy + h, sx : sx + w].copy()


class RandomUniformProvider:
    def __init__(self, low: float, high: float):
        self.low, self.high = low, high

    def get_value(self) -> float:
        return random.uniform(self.low, self.high)


class NoiseAugmentation:
    def __init__(
        self,
        std_provider: Optional[ParameterProviderProtocol] = None,
        prob_provider: Optional[ParameterProviderProtocol] = None,
    ):
        self.std_provider = std_provider or RandomUniformProvider(0.1, 5.0)
        self.prob_provider = prob_provider or RandomUniformProvider(0.01, 0.05)

    def gaussian_noise(
        self, M: np.ndarray, std: Optional[float] = None, **kwargs
    ) -> np.ndarray:
        actual_std = std if std is not None else self.std_provider.get_value()

        np_M = np.asanyarray(M, dtype=np.float32)
        noise = np.random.normal(0, actual_std, np_M.shape)
        return np.clip(np_M + noise, 0, 255).astype(np.uint8)

    def salt_and_pepper(
        self, M: np.ndarray, prob: Optional[float] = None, **kwargs
    ) -> np.ndarray:
        actual_prob = prob if prob is not None else self.prob_provider.get_value()

        result = np.copy(M).astype(np.uint8)
        random_map = np.random.random(result.shape)

        result[random_map < (actual_prob / 2)] = 0
        result[random_map > (1 - actual_prob / 2)] = 255
        return result


@apply_to_methods(auto_fill_color, ["erode", "get_boundaries", "morphology_filter"])
@apply_to_methods(kernel_data_processing, ["dilate", "erode", "get_boundaries"])
class MorphologyAugmentation:
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

        padded = np.pad(
            M_arr,
            pad_width=((pad_before, pad_after), (pad_before, pad_after)),
            mode="constant",
            constant_values=pad_value,
        )

        windows = sliding_window_view(padded, (kernel_size, kernel_size))

        if op_func is None:
            op_func = lambda win, axes: np.max(win, axis=axes)

        result = op_func(windows, (2, 3)).astype(np.uint8)
        return result[:h, :w]

    def dilate(self, M: np.ndarray, kernel_size: int, **kwargs) -> np.ndarray:
        return self._sliding_window_engine(M, kernel_size, pad_value=0)

    def erode(
        self, M: np.ndarray, kernel_size: int, fill: int = 0, **kwargs
    ) -> np.ndarray:
        return self._sliding_window_engine(
            M,
            kernel_size,
            op_func=lambda win, axes: np.min(win, axis=axes),
            pad_value=fill,
        )

    def get_boundaries(
        self, M: np.ndarray, kernel_size: int = 2, **kwargs
    ) -> np.ndarray:
        M_arr = np.asanyarray(M)

        eroded = self.erode(M_arr, kernel_size=kernel_size, **kwargs)

        diff = M_arr.astype(np.int16) - eroded.astype(np.int16)
        return np.clip(diff, 0, 255).astype(np.uint8)

    def morphology_filter(
        self,
        M: np.ndarray,
        kernel_size: int,
        fill: int = 0,
        mode: Literal["open", "close"] = "open",
        **kwargs,
    ) -> np.ndarray:
        if mode == "open":
            return self.dilate(self.erode(M, kernel_size, fill=fill), kernel_size)
        return self.erode(self.dilate(M, kernel_size), kernel_size, fill=fill)
