from typing import Protocol, TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray
MtxList: TypeAlias = list[np.ndarray]


class FeatureExtractorProtocol(Protocol):
    def extract_edges(self, matrix: Mtx) -> Mtx: ...
    def extract_features(self, matrix: Mtx) -> list[float]: ...


class FeatureExtraction:
    def __init__(self):
        pass

    def extract_edges(self, matrix: Mtx) -> Mtx:
        return matrix

    def extract_features(self, matrix: Mtx) -> list[float]:
        return [0.0]
