import pprint

import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score

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
        y_test = self.y_test
        y_pred = self.model.predict(self.X_test)

        if len(self.y_test.shape) > 1 and self.y_test.shape[1] > 1:
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
        cr = classification_report(y_test, y_pred)
        metrics = {
            "Confusion matrix": cr,
            "Recall": precision_score(y_test, y_pred),
            "Precision": recall_score(y_test, y_pred),
        }
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(metrics)
        return metrics

    def save_model(self):
        self.model.save()

    def load_model(self):
        self.model.load()
