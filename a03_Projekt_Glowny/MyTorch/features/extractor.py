from dataclasses import dataclass
from typing import List

from MyTorch import ImageGray

from .interfaces import FeatureExtractionABC


def extract_edges_logic(matrix: ImageGray) -> ImageGray:
    return matrix


def extract_features_vector_logic(matrix: ImageGray) -> List[float]:
    return list()


@dataclass(frozen=True, slots=True)
class FeatureExtraction(FeatureExtractionABC):
    def extract_edges(self, matrix: ImageGray) -> ImageGray:
        return extract_edges_logic(matrix)

    def extract_features(self, matrix: ImageGray) -> List[float]:
        return extract_features_vector_logic(matrix)
