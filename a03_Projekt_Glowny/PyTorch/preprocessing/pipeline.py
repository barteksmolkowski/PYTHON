import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, TypeAlias

import numpy as np
from common_utils import class_autologger
from preprocessing import (
    ConvolutionActions,
    ConvolutionProtocol,
    DataAugmentation,
    DataAugmentationProtocol,
    GrayScaleProcessing,
    GrayScaleProtocol,
    ImageConverterProtocol,
    ImageGeometry,
    ImageGeometryProtocol,
    ImageHandler,
    ImageHandlerProtocol,
    ImageToMatrixConverter,
    Normalization,
    NormalizationProtocol,
    Pooling,
    PoolingProtocol,
    Thresholding,
    ThresholdingProtocol,
    TransformPipeline,
)

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class TransformPipelineProtocol(Protocol):
    def apply(self, matrix: Mtx) -> MtxList: ...


class ImageDataPreprocessingProtocol(Protocol):
    def preprocess(self, path: str) -> Optional[list[MtxList]]: ...


def apply_pipeline_logic(
    matrix: np.ndarray,
    grayscale_func: Callable[[np.ndarray], np.ndarray],
    geometry_func: Callable[[np.ndarray, tuple[int, int]], np.ndarray],
    augment_func: Callable[[np.ndarray], list[np.ndarray]],
    normalize_func: Callable[[np.ndarray], np.ndarray],
) -> list[np.ndarray]:
    x = grayscale_func(matrix)
    x = geometry_func(x, (28, 28))
    augmented_samples = augment_func(x)
    return [normalize_func(sample) for sample in augmented_samples]


def preprocess_logic(
    channels: list[np.ndarray], apply_func: Callable[[np.ndarray], list[np.ndarray]]
) -> list[list[np.ndarray]]:
    return [apply_func(ch) for ch in channels]


@dataclass
@class_autologger
class TransformPipeline:
    geometry: Optional[ImageGeometryProtocol] = None
    normalization: Optional[NormalizationProtocol] = None
    grayscale: Optional[GrayScaleProtocol] = None
    thresholding: Optional[ThresholdingProtocol] = None
    convolution: Optional[ConvolutionProtocol] = None
    pooling: Optional[PoolingProtocol] = None
    augmentation: Optional[DataAugmentationProtocol] = None

    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.geometry = self.geometry or ImageGeometry()
        self.normalization = self.normalization or Normalization()
        self.grayscale = self.grayscale or GrayScaleProcessing()
        self.thresholding = self.thresholding or Thresholding()
        self.convolution = self.convolution or ConvolutionActions(
            geometry=self.geometry
        )
        self.pooling = self.pooling or Pooling()
        self.augmentation = self.augmentation or DataAugmentation()

    def apply(self, matrix: Mtx) -> MtxList:
        grayscale = self.grayscale
        geometry = self.geometry
        augmentation = self.augmentation
        normalization = self.normalization

        assert grayscale is not None
        assert geometry is not None
        assert augmentation is not None
        assert normalization is not None

        self.logger.debug(f"[apply] Starting pipeline for matrix shape={matrix.shape}")

        final_batch = apply_pipeline_logic(
            matrix=matrix,
            grayscale_func=lambda m: grayscale.convert_color_space(m, to_gray=True),
            geometry_func=lambda m, s: geometry.prepare_standard_geometry(
                m, target_size=s
            ),
            augment_func=lambda m: augmentation.augment(m),
            normalize_func=lambda m: normalization.process(m, use_z_score=True),
        )

        if not final_batch:
            self.logger.warning("[apply] Augmentation returned 0 samples.")
        else:
            self.logger.info(f"[apply] Validated {len(final_batch)} final matrices.")

        return final_batch


@dataclass
@class_autologger
class ImageDataPreprocessing:
    pipeline: Optional[TransformPipelineProtocol] = None

    handler: ImageHandlerProtocol = field(init=False, repr=False)
    converter: ImageConverterProtocol = field(init=False, repr=False)
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.handler = ImageHandler()
        self.converter = ImageToMatrixConverter()

        if self.pipeline is None:
            self.logger.debug(
                "[__post_init__] pipeline is None, selecting default TransformPipeline"
            )
            self.pipeline = TransformPipeline()

        self.logger.info(
            "[__post_init__] Validated initialization of ImageDataPreprocessing components."
        )

    def preprocess(self, path: str) -> Optional[list[MtxList]]:
        converter = self.converter
        pipeline = self.pipeline

        assert converter is not None
        assert pipeline is not None

        channels = converter.get_channels_from_file(path)

        if not channels:
            self.logger.error(
                f"[preprocess] Data loss: No channels extracted from path='{path}'"
            )
            return None

        self.logger.info(
            f"[preprocess] Validated {len(channels)} channels from path='{path}'."
        )

        processed_channels = preprocess_logic(
            channels=channels, apply_func=lambda ch: pipeline.apply(ch)
        )

        if len(processed_channels) != len(channels):
            self.logger.error(
                f"[preprocess] Pipeline mismatch: {len(processed_channels)} vs {len(channels)}"
            )

        self.logger.info(
            f"[preprocess] Validated {len(processed_channels)} processed batches."
        )
        return processed_channels
