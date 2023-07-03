import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.metrics import Precision, Recall
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from com.models.base_model import BaseModel


class TensorflowSequentialModel(BaseModel):
    """
    Tensorflow Sequential model
    """

    def __init__(
        self,
        input_dim: int = 574,
        output_dim: int = 2,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.3,
        random_state: int = 0,
    ):
        super().__init__()
        self.model = None
        self.input_dim = input_dim
        self.random_state = random_state
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

    def _f1_score(self, y_true, y_pred):
        """
        Custom F1 Score metrics
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (actual_positives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1

    def define(self) -> None:
        """
        Define the model
        """

        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

        self.model = Sequential(
            [
                Dense(64, input_shape=(self.input_dim,), activation="relu"),
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(self.output_dim, activation="sigmoid"),
            ]
        )

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(0.001),
            metrics=[Precision(), Recall()],
            run_eagerly=True,
        )
        self.model.summary()

    def train(self, X: np.array, y: np.array) -> dict:
        """
        Train the model.
        """
        history = self.model.fit(
            X, y, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.epochs
        )
        self.history = history

    def save(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.model.save(self.model_path)

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model folder does not exist")

        self.model = load_model(self.model_path)

    def create_dataset(self, df: pd.DataFrame) -> tuple:
        """Create dataset based on model"""
        Y = pd.get_dummies(df["has_accident"]).values

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
