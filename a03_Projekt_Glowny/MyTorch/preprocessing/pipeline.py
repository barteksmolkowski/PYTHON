from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from MyTorch import FilePath, ImageGray, ImageRGB, Shape
from preprocessing import (
    ConvolutionActions,
    DataAugmentation,
    GrayScaleProcessing,
    ImageGeometry,
    ImageHandler,
    ImageToMatrixConverter,
    Normalization,
    Pooling,
    Thresholding,
    TransformPipeline,
)

from .base import ImageDataPreprocessingABC, TransformPipelineABC
from .protocols import (
    ConvolutionProtocol,
    DataAugmentationProtocol,
    GrayScaleProtocol,
    ImageConverterProtocol,
    ImageGeometryProtocol,
    ImageHandlerProtocol,
    NormalizationProtocol,
    PoolingProtocol,
    ThresholdingProtocol,
    TransformPipelineProtocol,
)


def apply_pipeline_logic(
    matrix: ImageRGB,
    grayscale_func: Callable[[ImageRGB], ImageGray],
    geometry_func: Callable[[ImageGray, Shape], ImageGray],
    augment_func: Callable[[ImageGray], List[ImageGray]],
    normalize_func: Callable[[ImageGray], ImageGray],
) -> List[ImageGray]:
    x_gray: ImageGray = grayscale_func(matrix)
    x_geom: ImageGray = geometry_func(x_gray, (28, 28))
    augmented_samples: List[ImageGray] = augment_func(x_geom)
    return [normalize_func(sample) for sample in augmented_samples]


def preprocess_logic(
    channels: List[ImageGray], apply_func: Callable[[ImageGray], List[ImageGray]]
) -> List[List[ImageGray]]:
    return [apply_func(ch) for ch in channels]


@dataclass(frozen=True, slots=True)
class TransformPipeline(TransformPipelineABC):
    geometry: ImageGeometryProtocol = field(default_factory=ImageGeometry, repr=False)
    normalization: NormalizationProtocol = field(
        default_factory=Normalization, repr=False
    )
    grayscale: GrayScaleProtocol = field(
        default_factory=GrayScaleProcessing, repr=False
    )
    thresholding: ThresholdingProtocol = field(default_factory=Thresholding, repr=False)
    convolution: ConvolutionProtocol = field(
        default_factory=ConvolutionActions, repr=False
    )
    pooling: PoolingProtocol = field(default_factory=Pooling, repr=False)
    augmentation: DataAugmentationProtocol = field(
        default_factory=DataAugmentation, repr=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()

    def apply(self, matrix: ImageRGB) -> List[ImageGray]:
        return apply_pipeline_logic(
            matrix=matrix,
            grayscale_func=lambda m: self.grayscale.convert_color_space(
                m, to_gray=True
            ),
            geometry_func=lambda m, s: self.geometry.prepare_standard_geometry(
                m, target_size=s
            ),
            augment_func=lambda m: self.augmentation.augment(m),
            normalize_func=lambda m: self.normalization.process(m, use_z_score=True),
        )


@dataclass(frozen=True, slots=True)
class ImageDataPreprocessing(ImageDataPreprocessingABC):
    pipeline: TransformPipelineProtocol = field(default_factory=TransformPipeline)
    handler: ImageHandlerProtocol = field(default_factory=ImageHandler, repr=False)
    converter: ImageConverterProtocol = field(
        default_factory=ImageToMatrixConverter, repr=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()

    def preprocess(self, path: FilePath) -> List[List[ImageGray]]:
        channels: List[ImageGray] = self.converter.get_channels_from_file(path)

        def _apply_pipeline(ch: ImageGray) -> List[ImageGray]:
            matrix_rgb: ImageRGB = np.stack([ch] * 3, axis=0).astype(np.float32)
            return self.pipeline.apply(matrix_rgb)

        return preprocess_logic(channels=channels, apply_func=_apply_pipeline)
