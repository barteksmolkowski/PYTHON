from typing import TypeAlias

import numpy as np

Mtx: TypeAlias = np.ndarray


class Trainer:
    def __init__(self) -> None:
        pass

    def train(self, epochs: int) -> None:
        pass

    def evaluate(self) -> None:
        pass
