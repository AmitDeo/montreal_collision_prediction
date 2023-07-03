import os
from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """
    Base model that defined the method to be implemented
    by the child model.
    """

    MODEL_PATH = ""

    def __init__(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    @property
    def model_path(self):
        self.MODEL_PATH = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "saved_models",
                os.path.splitext(self.__class__.__name__.lower())[0],
            )
        )
        return self.MODEL_PATH

    @abstractmethod
    def define() -> None:
        """
        Define the model
        """

    @abstractmethod
    def train(self, X: np.array, y: np.array) -> None:
        """
        Train the model.
        """

    @abstractmethod
    def save(self):
        """
        Save the model
        """

    @abstractmethod
    def load(self):
        """
        Load the model
        """

    @abstractmethod
    def predict(self):
        """
        Predict the model
        """
