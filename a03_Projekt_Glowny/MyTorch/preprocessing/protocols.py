from typing import List, Literal, Protocol, Tuple, Union, overload

from MyTorch import T2D, FilePath, ImageGray, ImageRGB, JsonData, Padded, Shape


class DataAugmentationProtocol(Protocol):
    def augment(
        self, M: ImageGray, repeats: int = 1, debug: bool = False
    ) -> List[ImageGray]: ...


class GeometryAugmentationProtocol(Protocol):
    def horizontal_flip(self, M: ImageGray) -> ImageGray: ...

    def vertical_flip(self, M: ImageGray) -> ImageGray: ...

    def rotate_90(self, M: ImageGray, is_right: bool = True) -> ImageGray: ...

    def rotate_small_angle(
        self,
        M: ImageGray,
        h: int,
        w: int,
        params: JsonData,
        angle: float = 0.0,
        fill: int = 0,
    ) -> ImageGray: ...

    def random_shift(
        self,
        M: ImageGray,
        h: int,
        w: int,
        fill: int = 0,
        is_right: bool = True,
    ) -> ImageGray: ...


class ParameterProviderProtocol(Protocol):
    def get_value(self) -> float: ...


class NoiseAugmentationProtocol(Protocol):
    def gaussian_noise(self, M: ImageGray, std: float = 0.0) -> ImageGray: ...

    def salt_and_pepper(self, M: ImageGray, prob: float = 0.0) -> ImageGray: ...


class MorphologyAugmentationProtocol(Protocol):
    def dilate(self, M: ImageGray, kernel_size: int) -> ImageGray: ...

    def erode(self, M: ImageGray, kernel_size: int, fill: int = 0) -> ImageGray: ...

    def get_boundaries(self, M: ImageGray, kernel_size: int = 2) -> ImageGray: ...

    def morphology_filter(
        self, M: ImageGray, kernel_size: int, fill: int = 0, mode: str = "open"
    ) -> ImageGray: ...


class ImageConverterProtocol(Protocol):
    def get_channels_from_file(self, path: FilePath) -> List[ImageGray]: ...


class ConvolutionProtocol(Protocol):
    def convolution_2d(self, M: ImageGray, filters: List[T2D]) -> List[ImageGray]: ...

    def apply_filters(
        self, channels: List[ImageGray], filters: List[T2D]
    ) -> List[ImageGray]: ...


class ImageGeometryProtocol(Protocol):
    def resize(self, M: ImageGray, new_size: Shape) -> ImageGray: ...

    def pad(self, M: ImageGray, pad_value: int, padding: int) -> Padded: ...

    def prepare_standard_geometry(
        self, M: ImageGray, target_size: Shape, padding: int = 2, pad_value: int = 0
    ) -> Padded: ...


class GrayScaleProtocol(Protocol):
    @overload
    def convert_color_space(
        self, M: ImageRGB, to_gray: Literal[True] = True
    ) -> ImageGray: ...

    @overload
    def convert_color_space(
        self, M: ImageGray, to_gray: Literal[False]
    ) -> ImageRGB: ...

    def convert_color_space(
        self, M: Union[ImageGray, ImageRGB], to_gray: bool = True
    ) -> Union[ImageGray, ImageRGB]: ...


class ImageHandlerProtocol(Protocol):
    def open_image(self, path: FilePath) -> Tuple[ImageRGB, int, int]: ...

    def save(self, data: Union[ImageGray, ImageRGB], path: FilePath) -> None: ...

    @overload
    def handle_file(
        self,
        path: FilePath,
        data: Union[ImageGray, ImageRGB],
        is_save_mode: Literal[True],
    ) -> Union[ImageGray, ImageRGB]: ...

    @overload
    def handle_file(
        self, path: FilePath, data: ImageGray, is_save_mode: Literal[False] = False
    ) -> ImageRGB: ...

    def handle_file(
        self,
        path: FilePath,
        data: Union[ImageGray, ImageRGB],
        is_save_mode: bool = False,
    ) -> Union[ImageGray, ImageRGB]: ...


class NormalizationProtocol(Protocol):
    def normalize(
        self,
        M: ImageGray,
        old_r: Tuple[float, float] = (0.0, 255.0),
        new_r: Tuple[float, float] = (0.0, 1.0),
    ) -> ImageGray: ...

    def z_score_normalization(self, M: ImageGray) -> ImageGray: ...

    @overload
    def process(self, M: ImageGray, use_z_score: Literal[True]) -> ImageGray: ...

    @overload
    def process(
        self,
        M: ImageGray,
        use_z_score: Literal[False],
        old_r: Tuple[float, float],
        new_r: Tuple[float, float],
    ) -> ImageGray: ...

    def process(
        self,
        M: ImageGray,
        use_z_score: bool = True,
        old_r: Tuple[float, float] = (0.0, 255.0),
        new_r: Tuple[float, float] = (0.0, 1.0),
    ) -> ImageGray: ...


class TransformPipelineProtocol(Protocol):
    def apply(self, matrix: ImageRGB) -> List[ImageGray]: ...


class ImageDataPreprocessingProtocol(Protocol):
    def preprocess(self, path: FilePath) -> List[List[ImageGray]]: ...


class PoolingProtocol(Protocol):
    def max_pool(
        self,
        matrix: ImageGray,
        kernel_size: Shape = (2, 2),
        stride: int = 0,
        pad_width: int = 0,
        pad_values: int = 0,
    ) -> ImageGray: ...


class ThresholdingProtocol(Protocol):
    @overload
    def adaptive_threshold(
        self,
        matrix: ImageGray,
        block_size: int = 5,
        c: int = 2,
        auto_params: Literal[True] = True,
    ) -> ImageGray: ...

    @overload
    def adaptive_threshold(
        self, matrix: ImageGray, block_size: int, c: int, auto_params: Literal[False]
    ) -> ImageGray: ...

    def adaptive_threshold(
        self,
        matrix: ImageGray,
        block_size: int = 5,
        c: int = 2,
        auto_params: bool = True,
    ) -> ImageGray: ...
