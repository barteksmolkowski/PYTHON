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
    def preprocess(self, path: str) -> Optional[MtxList]: ...


@class_autologger
class TransformPipeline:
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
        self.geometry = geometry or ImageGeometry()
        self.normalization = normalization or Normalization()
        self.grayscale = grayscale or GrayScaleProcessing()
        self.thresholding = thresholding or Thresholding()
        self.convolution = convolution or ConvolutionActions(geometry=self.geometry)
        self.pooling = pooling or Pooling()
        self.augmentation = augmentation or DataAugmentation()

    def apply(self, matrix: Mtx) -> MtxList:
        x = self.grayscale.convert_color_space(matrix, to_gray=True)
        self.logger.debug(f"[apply] Initial grayscale conversion complete. Shape: {x.shape}")

        x = self.geometry.prepare_standard_geometry(x, target_size=(28, 28))
        self.logger.debug(f"[apply] Geometry standardized to 28x28 target size.")

        augmented_samples = self.augmentation.augment(x)
        sample_count = len(augmented_samples)
        
        if sample_count == 0:
            self.logger.warning(f"[apply] Augmentation returned 0 samples for the input matrix.")
        else:
            self.logger.info(f"[apply] Generated {sample_count} augmented samples for processing.")

        final_batch = []
        for i, sample in enumerate(augmented_samples):
            normalized = self.normalization.process(sample, use_z_score=True)
            final_batch.append(normalized)
            
            if (i + 1) % 5 == 0 or (i + 1) == sample_count:
                self.logger.debug(f"[apply] Progress: Normalized {i + 1}/{sample_count} samples.")

        self.logger.info(f"[apply] Pipeline execution finished. Returning batch of {len(final_batch)} matrices.")
        return final_batch


@class_autologger
class ImageDataPreprocessing:
    def __init__(self, pipeline: Optional[TransformPipeline] = None):
        self.handler = ImageHandler()
        self.converter = ImageToMatrixConverter()
        self.pipeline = pipeline or TransformPipeline()

    def preprocess(self, path: str) -> Optional[MtxList]:
        try:
            channels = self.converter.get_channels_from_file(path)
            
            self.logger.info(f"[preprocess] Successfully loaded {len(channels)} channels from {path}. Starting pipeline transformation.")
            
            processed_channels = [self.pipeline.apply(ch) for ch in channels]
            
            self.logger.debug(f"[preprocess] Pipeline applied to all {len(processed_channels)} channels.")
            return processed_channels

        except Exception as e:
            self.logger.error(f"[preprocess] Preprocessing failed for file {path}. Error details: {str(e)}")
            print(f"[CRITICAL] Preprocessing Error: {e}")
            return None
