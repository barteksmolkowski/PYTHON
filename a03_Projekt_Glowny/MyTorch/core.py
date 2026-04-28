import logging

from common_utils import class_autologger
from nn import NeuralNetwork
from preprocessing import ImageDataPreprocessing
from training import Trainer


@class_autologger
class BrainEngine:
    logger: logging.Logger

    def __init__(self, preprocessor=None, nn=None, trainer=None) -> None:
        if preprocessor is None:
            self.logger.debug(
                "[__init__] preprocessor is None, using default ImageDataPreprocessing"
            )
        self.preprocessor = preprocessor or ImageDataPreprocessing()

        if nn is None:
            self.logger.debug(
                "[__init__] nn is None, initializing NeuralNetwork with None model"
            )
            nn = NeuralNetwork(model=None)

        self.nn = nn

        if trainer is None:
            self.logger.debug("[__init__] trainer is None, using default Trainer")
        self.trainer = trainer or Trainer()

    def run_training(self, epochs: int = 10) -> None:
        self.logger.info(f"Starting training process: {epochs} epochs.")

    def recognize_digit(self, path: str) -> None:
        self.logger.info(f"Starting digit recognition for path: {path}")
