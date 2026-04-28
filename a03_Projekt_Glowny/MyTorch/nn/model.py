from dataclasses import dataclass, field
from typing import List

import numpy as np
from interfaces import ModelABC, OrchestratorABC

from MyTorch import T4D, FilePath
from nn import LayerProtocol, Sequential


def chain_fwd(layers: List[LayerProtocol], x: T4D) -> T4D:
    return np.array([], dtype=np.float32).reshape(0, 0, 0, 0)


def chain_bwd(layers: List[LayerProtocol], grad: T4D) -> T4D:
    return np.array([], dtype=np.float32).reshape(0, 0, 0, 0)


def add_layer(layers: List[LayerProtocol], layer: LayerProtocol) -> None:
    pass


def predict(model: Sequential, x: T4D) -> T4D:
    return np.array([], dtype=np.float32).reshape(0, 0, 0, 0)


def save(model: Sequential, path: FilePath) -> None:
    """save zrobić w .npz"""


def load(model: Sequential, path: FilePath) -> None:
    """load zrobić w .npz"""


@dataclass(frozen=True, slots=True)
class Sequential(OrchestratorABC):
    layers: List[LayerProtocol] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def add(self, layer: LayerProtocol) -> None:
        add_layer(self.layers, layer)

    def forward(self, x: T4D) -> T4D:
        return chain_fwd(self.layers, x)

    def backward(self, grad: T4D) -> T4D:
        return chain_bwd(self.layers, grad)


@dataclass(frozen=True, slots=True)
class NeuralNetwork(ModelABC):
    model: Sequential = field(default_factory=Sequential)

    def __post_init__(self) -> None:
        super().__post_init__()

    def predict(self, x: T4D) -> T4D:
        return predict(self.model, x)

    def save(self, path: FilePath) -> None:
        self.logger.info(f"Saving model to {path}")
        save(self.model, path)

    def load(self, path: FilePath) -> None:
        self.logger.info(f"Loading model from {path}")
        load(self.model, path)
