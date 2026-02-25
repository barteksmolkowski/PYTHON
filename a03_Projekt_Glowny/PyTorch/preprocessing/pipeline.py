import logging
from typing import Optional, Protocol, TypeAlias

import numpy as np

from common_utils import class_autologger

from .augmentation import DataAugmentation
from .conversion import ImageToMatrixConverter
from .convolution import ConvolutionActions
from .geometry import ImageGeometry
from .grayscale import GrayScaleProcessing
from .io_image import ImageHandler
from .normalization import Normalization
from .pooling import Pooling
from .thresholding import Thresholding

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class TransformPipelineProtocol(Protocol):
    def apply(self, matrix: Mtx) -> MtxList: ...


class ImageDataPreprocessingProtocol(Protocol):
    def preprocess(self, path: str) -> Optional[list[MtxList]]: ...


@class_autologger
class TransformPipeline:
    logger: logging.Logger

    def __init__(
        self,
        geometry: Optional[ImageGeometry] = None,
        normalization: Optional[Normalization] = None,
        grayscale: Optional[GrayScaleProcessing] = None,
        thresholding: Optional[Thresholding] = None,
        convolution: Optional[ConvolutionActions] = None,
        pooling: Optional[Pooling] = None,
        augmentation: Optional[DataAugmentation] = None,
    ):
        if geometry is None:
            self.logger.debug(
                "[__init__] geometry is None, selecting default ImageGeometry"
            )
        self.geometry = geometry or ImageGeometry()

        if normalization is None:
            self.logger.debug(
                "[__init__] normalization is None, selecting default Normalization"
            )
        self.normalization = normalization or Normalization()

        if grayscale is None:
            self.logger.debug(
                "[__init__] grayscale is None, selecting default GrayScaleProcessing"
            )
        self.grayscale = grayscale or GrayScaleProcessing()

        if thresholding is None:
            self.logger.debug(
                "[__init__] thresholding is None, selecting default Thresholding"
            )
        self.thresholding = thresholding or Thresholding()

        if convolution is None:
            self.logger.debug(
                "[__init__] convolution is None, selecting default ConvolutionActions"
            )
        self.convolution = convolution or ConvolutionActions(geometry=self.geometry)

        if pooling is None:
            self.logger.debug("[__init__] pooling is None, selecting default Pooling")
        self.pooling = pooling or Pooling()

        if augmentation is None:
            self.logger.debug(
                "[__init__] augmentation is None, selecting default DataAugmentation"
            )
        self.augmentation = augmentation or DataAugmentation()

        self.logger.info(
            "[__init__] Validated initialization of all pipeline components."
        )

    def apply(self, matrix: Mtx) -> MtxList:
        x = self.grayscale.convert_color_space(matrix, to_gray=True)
        self.logger.debug(
            f"[apply] Initial grayscale conversion complete: shape={x.shape}"
        )

        x = self.geometry.prepare_standard_geometry(x, target_size=(28, 28))
        self.logger.debug(f"[apply] Geometry standardized to target_size=(28, 28)")

        augmented_samples = self.augmentation.augment(x)
        sample_count = len(augmented_samples)

        if sample_count == 0:
            self.logger.warning(
                f"[apply] Logic error: Augmentation returned 0 samples for input_shape={x.shape}"
            )
        else:
            self.logger.info(
                f"[apply] Validated {sample_count} augmented samples for processing."
            )

        final_batch = []
        for i, sample in enumerate(augmented_samples):
            normalized = self.normalization.process(sample, use_z_score=True)
            final_batch.append(normalized)

            if (i + 1) % 5 == 0 or (i + 1) == sample_count:
                self.logger.debug(
                    f"[apply] Batch progress: normalized={i + 1}, total_expected={sample_count}"
                )

        if len(final_batch) != sample_count:
            self.logger.error(
                f"[apply] Data loss: processed={len(final_batch)} vs expected={sample_count}"
            )

        self.logger.info(
            f"[apply] Validated {len(final_batch)} final matrices in batch."
        )
        return final_batch


@class_autologger
class ImageDataPreprocessing:
    logger: logging.Logger

    def __init__(self, pipeline: Optional[TransformPipeline] = None):
        self.handler = ImageHandler()
        self.converter = ImageToMatrixConverter()

        if pipeline is None:
            self.logger.debug(
                "[__init__] pipeline is None, selecting default TransformPipeline"
            )
        self.pipeline = pipeline or TransformPipeline()

        self.logger.info(
            "[__init__] Validated initialization of ImageDataPreprocessing components."
        )

    def preprocess(self, path: str) -> Optional[list[MtxList]]:
        channels = self.converter.get_channels_from_file(path)

        if not channels:
            self.logger.error(
                f"[preprocess] Data loss: No channels extracted from path='{path}'"
            )
            return None

        self.logger.info(
            f"[preprocess] Validated {len(channels)} channels from path='{path}'. Starting pipeline transformation."
        )

        processed_channels: list[MtxList] = [self.pipeline.apply(ch) for ch in channels]

        if len(processed_channels) != len(channels):
            self.logger.error(
                f"[preprocess] Pipeline mismatch: processed_count={len(processed_channels)} vs original_count={len(channels)}"
            )

        self.logger.info(
            f"[preprocess] Validated {len(processed_channels)} processed channel batches."
        )
        return processed_channels
