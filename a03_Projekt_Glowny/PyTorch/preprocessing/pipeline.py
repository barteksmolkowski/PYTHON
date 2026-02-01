from typing import Optional, Protocol, TypeAlias

import numpy as np

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
        x = self.geometry.prepare_standard_geometry(x, target_size=(28, 28))

        augmented_samples = self.augmentation.augment(x)

        final_batch = [
            self.normalization.process(sample, use_z_score=True)
            for sample in augmented_samples
        ]
        return final_batch


class ImageDataPreprocessing:
    def __init__(self, pipeline: Optional[TransformPipeline] = None):
        self.handler = ImageHandler()
        self.converter = ImageToMatrixConverter()
        self.pipeline = pipeline or TransformPipeline()

    def preprocess(self, path: str) -> Optional[MtxList]:
        try:
            channels = self.converter.get_channels_from_file(path)
            return [self.pipeline.apply(ch) for ch in channels]

        except Exception as e:
            print(f"[CRITICAL] Preprocessing Error: {e}")
            return None
