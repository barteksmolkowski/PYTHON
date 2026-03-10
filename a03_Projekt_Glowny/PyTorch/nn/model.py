import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from common_utils import class_autologger
from nn import LayerProtocol, Mtx, Sequential, Tensor, validate_shape


def execute_forward_chain(layers: List[LayerProtocol], x: Mtx) -> Mtx: ...


def execute_backward_chain(layers: List[LayerProtocol], grad: Mtx) -> Mtx: ...


def insert_layer_into_stack(
    layers: List[LayerProtocol], layer: LayerProtocol
) -> None: ...


def run_network_prediction(model: Sequential, x: Mtx) -> Mtx: ...


def persist_network_state(model: Sequential, path: str) -> None: ...


def restore_network_state(model: Sequential, path: str) -> None: ...


@class_autologger
@dataclass
class Sequential:
    layers: List[LayerProtocol] = field(default_factory=list)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def add(self, layer: LayerProtocol) -> None:
        insert_layer_into_stack(self.layers, layer)

    def forward(self, x: Mtx) -> Mtx:
        return execute_forward_chain(self.layers, x)

    def backward(self, grad: Mtx) -> Mtx:
        return execute_backward_chain(self.layers, grad)


@class_autologger
@dataclass
class NeuralNetwork:
    model: Sequential = field(default_factory=Sequential)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict(self, x: Mtx) -> Mtx:
        return run_network_prediction(self.model, x)

    def save(self, path: str) -> None:
        """save zrobić w .npz"""
        persist_network_state(self.model, path)

    def load(self, path: str) -> None:
        """load zrobić w .npz"""
        restore_network_state(self.model, path)
