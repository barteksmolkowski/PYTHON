from nn.model import NeuralNetwork
from preprocessing.pipeline import ImageDataPreprocessing
from training.trainer import Trainer

""" podzielić foldery na logiczne całości żeby było jak najmniej skomplikowanych połączeń """


class BrainEngine:
    def __init__(self):
        self.preprocessor = ImageDataPreprocessing()
        self.nn = NeuralNetwork()
        self.trainer = Trainer()

    def run_training(self):
        pass

    def recognize_digit(self, path):
        pass
