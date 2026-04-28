from dataclasses import dataclass, field
from typing import List

from interfaces import TrainerABC

from MyTorch import BatchData, LayerProtocol, LossProtocol, OptimizerProtocol


@dataclass(frozen=False, slots=True)
class Trainer(TrainerABC):
    model: LayerProtocol
    optimizer: OptimizerProtocol
    loss_fn: LossProtocol
    train_loader: BatchData = field(default_factory=list)
    test_loader: BatchData = field(default_factory=list)
    history: List[float] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()

    def train(self, epochs: int) -> None:
        for epoch in range(epochs):
            self.history.append(0.0)

    def evaluate(self) -> float:
        return 0.0
