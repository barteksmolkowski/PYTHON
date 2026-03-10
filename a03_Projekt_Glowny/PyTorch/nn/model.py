import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from common_utils import class_autologger
from nn import LayerProtocol, Mtx, Sequential, T, validate_shape


def chain_fwd(layers: List[LayerProtocol], x: Mtx) -> Mtx: ...


def chain_bwd(layers: List[LayerProtocol], grad: Mtx) -> Mtx: ...


def add_layer(layers: List[LayerProtocol], layer: LayerProtocol) -> None: ...


def predict(model: Sequential, x: Mtx) -> Mtx: ...


def save(model: Sequential, path: str) -> None: ...


def load(model: Sequential, path: str) -> None: ...


@class_autologger
@dataclass
class Sequential:
    layers: List[LayerProtocol] = field(default_factory=list)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def add(self, layer: LayerProtocol) -> None:
        add_layer(self.layers, layer)

    def forward(self, x: Mtx) -> Mtx:
        return chain_fwd(self.layers, x)

    def backward(self, grad: Mtx) -> Mtx:
        return chain_bwd(self.layers, grad)


@class_autologger
@dataclass
class NeuralNetwork:
    model: Sequential = field(default_factory=Sequential)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict(self, x: Mtx) -> Mtx:
        return predict(self.model, x)

    def save(self, path: str) -> None:
        """save zrobić w .npz"""
        save(self.model, path)

    def load(self, path: str) -> None:
        """load zrobić w .npz"""
        load(self.model, path)
