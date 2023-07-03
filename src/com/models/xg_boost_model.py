import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from com.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XG Boot Model
    """

    def __init__(
        self,
        objective: str = "binary:logistic",
        random_state: int = 0,
        verbose: int = 0,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.01,
    ):
        super().__init__()

        self.model = None
        self.objective = objective
        self.random_state = random_state
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def define(self) -> None:
        """
        Define the model
        """
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self.objective,
            random_state=self.random_state,
        )

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Train the model.
        """
        self.model.fit(X, y)

    def save(self):
        self.model.save_model(f"{self.model_path}/xgb_model.json")

    def load(self):
        model_file = f"{self.model_path}/xgb_model.json"
        if not os.path.exists(model_file):
            raise FileNotFoundError("Saved model not found")

        self.model.load_model(model_file)

    def create_dataset(self, df: pd.DataFrame) -> tuple:
        """Create dataset based on model"""
        Y = df["has_accident"]
        X = df.drop(
            labels=[
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
