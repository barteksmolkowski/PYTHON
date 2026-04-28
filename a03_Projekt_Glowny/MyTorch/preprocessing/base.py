import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Union

from MyTorch import T2D, FilePath, ImageGray, ImageRGB, JsonData, Padded, Shape


@dataclass(frozen=True, slots=True)
class GeometryABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def horizontal_flip(self, M: ImageGray) -> ImageGray: ...

    @abstractmethod
    def vertical_flip(self, M: ImageGray) -> ImageGray: ...

    @abstractmethod
    def rotate_90(self, M: ImageGray, is_right: bool = True) -> ImageGray: ...

    @abstractmethod
    def rotate_small_angle(
        self,
        M: ImageGray,
        h: int,
        w: int,
        params: JsonData,
        angle: float = 0.0,
        fill: int = 0,
    ) -> ImageGray: ...

    @abstractmethod
    def random_shift(
        self,
        M: ImageGray,
        h: int,
        w: int,
        fill: int = 0,
        is_right: bool = True,
    ) -> ImageGray: ...


@dataclass(frozen=True, slots=True)
class ProviderABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def get_value(self) -> float: ...


@dataclass(frozen=True, slots=True)
class NoiseABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def gaussian_noise(self, M: ImageGray, std: float = 0.0) -> ImageGray: ...

    @abstractmethod
    def salt_and_pepper(self, M: ImageGray, prob: float = 0.0) -> ImageGray: ...


@dataclass(frozen=True, slots=True)
class MorphologyABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def dilate(self, M: ImageGray, kernel_size: int) -> ImageGray: ...

    @abstractmethod
    def erode(self, M: ImageGray, kernel_size: int, fill: int = 0) -> ImageGray: ...

    @abstractmethod
    def get_boundaries(self, M: ImageGray, kernel_size: int = 2) -> ImageGray: ...

    @abstractmethod
    def morphology_filter(
        self, M: ImageGray, kernel_size: int, fill: int = 0, mode: str = "open"
    ) -> ImageGray: ...


@dataclass(frozen=True, slots=True)
class AugmentationABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def augment(
        self, M: ImageGray, repeats: int = 1, debug: bool = False
    ) -> List[ImageGray]: ...


@dataclass(frozen=True, slots=True)
class ConverterABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def get_channels_from_file(self, path: FilePath) -> List[ImageGray]: ...


@dataclass(frozen=True, slots=True)
class ConvolutionABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def convolution_2d(self, M: ImageGray, filters: List[T2D]) -> List[ImageGray]: ...

    @abstractmethod
    def apply_filters(
        self, channels: List[ImageGray], filters: List[T2D]
    ) -> List[ImageGray]: ...


@dataclass(frozen=True, slots=True)
class ImageGeometryABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def resize(self, M: ImageGray, new_size: Shape) -> ImageGray: ...

    @abstractmethod
    def pad(self, M: ImageGray, pad_value: int, padding: int) -> Padded: ...

    @abstractmethod
    def prepare_standard_geometry(
        self, M: ImageGray, target_size: Shape, padding: int = 2, pad_value: int = 0
    ) -> Padded: ...


@dataclass(frozen=True, slots=True)
class GrayScaleABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def convert_color_space(
        self, M: Union[ImageGray, ImageRGB], to_gray: bool = True
    ) -> Union[ImageGray, ImageRGB]: ...


@dataclass(frozen=True, slots=True)
class ImageHandlerABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def open_image(self, path: FilePath) -> Tuple[ImageRGB, int, int]: ...

    @abstractmethod
    def save(self, data: Union[ImageGray, ImageRGB], path: FilePath) -> None: ...

    @abstractmethod
    def handle_file(
        self,
        path: FilePath,
        data: Union[ImageGray, ImageRGB],
        is_save_mode: bool = False,
    ) -> Union[ImageGray, ImageRGB]: ...


@dataclass(frozen=True, slots=True)
class NormalizationABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def normalize(
        self,
        M: ImageGray,
        old_r: Tuple[float, float] = (0.0, 255.0),
        new_r: Tuple[float, float] = (0.0, 1.0),
    ) -> ImageGray: ...

    @abstractmethod
    def z_score_normalization(self, M: ImageGray) -> ImageGray: ...

    @abstractmethod
    def process(
        self,
        M: ImageGray,
        use_z_score: bool = True,
        old_r: Tuple[float, float] = (0.0, 255.0),
        new_r: Tuple[float, float] = (0.0, 1.0),
    ) -> ImageGray: ...


@dataclass(frozen=True, slots=True)
class TransformPipelineABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def apply(self, matrix: ImageRGB) -> List[ImageGray]: ...


@dataclass(frozen=True, slots=True)
class ImageDataPreprocessingABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def preprocess(self, path: FilePath) -> List[List[ImageGray]]: ...


@dataclass(frozen=True, slots=True)
class PoolingABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def max_pool(
        self,
        matrix: ImageGray,
        kernel_size: Shape = (2, 2),
        stride: int = 0,
        pad_width: int = 0,
        pad_values: int = 0,
    ) -> ImageGray: ...


@dataclass(frozen=True, slots=True)
class ThresholdingABC(ABC):
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "logger", logging.getLogger(self.__class__.__name__))

    @abstractmethod
    def adaptive_threshold(
        self,
        matrix: ImageGray,
        block_size: int = 5,
        c: int = 2,
        auto_params: bool = True,
    ) -> ImageGray: ...
