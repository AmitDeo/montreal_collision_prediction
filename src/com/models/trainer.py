import pandas as pd
from sklearn.metrics import classification_report

from com.models.base_model import BaseModel


class Trainer:
    """
    Train model based on model provided.
    """

    def __init__(self, model: BaseModel, dataframe: pd.DataFrame):
        self.model = model
        self.model.define()
        self.dataframe = dataframe
        self.X_train, self.X_test, self.y_train, self.y_test = self.model.create_dataset(
            self.dataframe
        )

    def train_model(self):
        """
        Train the model
        """
        self.model.train(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        cr = classification_report(self.y_test, y_pred)
        return cr

    def save_model(self):
        self.model.save()

    def load_model(self):
        self.model.load()
