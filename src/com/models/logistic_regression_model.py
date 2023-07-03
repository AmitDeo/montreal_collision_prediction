import os
import pickle

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from com.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Logistic regression model
    """

    def __init__(
        self,
        penalty: str = "l1",
        solver: str = "liblinear",
        random_state: int = 0,
        verbose: int = 0,
    ):
        super().__init__()

        self.model = None
        self.penalty = penalty
        self.random_state = random_state
        self.verbose = verbose
        self.solver = solver

    def define(self) -> None:
        """
        Define the model
        """
        self.model = LogisticRegression(
            penalty=self.penalty, solver=self.solver, verbose=self.verbose
        )

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Train the model.
        """
        self.model.fit(X, y)

    def save(self):
        with open(f"{self.model_path}/model_pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model folder does not exist")

        with open(f"{self.model_path}/model_pkl", "wb") as f:
            self.model = pickle.load(self.model, f)

    def create_dataset(self, df: pd.DataFrame) -> tuple:
        """Create dataset based on model"""
        Y = df["has_accident"]
        X = df.drop(
            labels=[
                "Unnamed: 0",
                "date_accdn",
                "has_accident",
                "GridName",
                "number_comments",
                "number_complaints",
                "number_requests",
                "date_of_incident",
                "grid_area",
                "grid_long",
                "grid_lat",
                "number_of_accident_hour",
                "rues_accdn",
            ],
            axis=1,
        )

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        scaler = MinMaxScaler()
        X_train_normalized = scaler.fit_transform(X_train)

        # Save scaler for predict
        joblib.dump(scaler, f"{self.model_path}/scaler")

        return X_train_normalized, X_test, y_train, y_test

    def predict(self, X):
        scaler = joblib.load(f"{self.model_path}/scaler")
        X_transformed = scaler.transform(X)
        return self.model.predict(X_transformed)
