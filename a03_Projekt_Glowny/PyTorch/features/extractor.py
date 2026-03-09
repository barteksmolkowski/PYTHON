import logging
from dataclasses import dataclass, field
from typing import List, Protocol, TypeAlias

import numpy as np
from common_utils import class_autologger, silent

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class FeatureExtractorProtocol(Protocol):
    def extract_edges(self, matrix: Mtx) -> Mtx: ...
    def extract_features(self, matrix: Mtx) -> list[float]: ...


def extract_edges_logic(matrix: Mtx) -> Mtx:
    return matrix


def extract_features_vector_logic(matrix: Mtx) -> List[float]:
    return [0.0]


@class_autologger
@dataclass
class FeatureExtraction:
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_edges(self, matrix: Mtx) -> Mtx:
        self.logger.debug(f"Extracting edges from matrix shape: {matrix.shape}")
        return extract_edges_logic(matrix)

    @silent
    def extract_features(self, matrix: Mtx) -> List[float]:
        return extract_features_vector_logic(matrix)
