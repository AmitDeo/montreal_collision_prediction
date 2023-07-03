from com.models.base_model import BaseModel
from com.models.logistic_regression_model import LogisticRegressionModel
from com.models.tensorflow_sequential import TensorflowSequentialModel
from com.models.xg_boost_model import XGBoostModel


def select_model(model_name: str) -> BaseModel:
    """
    Select model based on string
    """
    if model_name == "logistic":
        model = LogisticRegressionModel(verbose=1)
    elif model_name == "sequential":
        model = TensorflowSequentialModel(epochs=5)
    elif model_name == "xgboost":
        model = XGBoostModel()

    return model
