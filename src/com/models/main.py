import typer

from com.models.logistic_regression_model import LogisticRegressionModel
from com.models.tensorflow_sequential import TensorflowSequentialModel
from com.models.trainer import Trainer
from com.models.utils.load_data import load_data

app = typer.Typer()


@app.command()
def train(model_name: str, data: str = "cleaned_data.csv"):
    """
    The function to train the model as defined in model
    """

    model = None
    if model_name == "logistic":
        model = LogisticRegressionModel(verbose=1)
    elif model_name == "sequential":
        model = TensorflowSequentialModel(epochs=5)

    df = load_data("cleaned_data.csv")
    trainer = Trainer(model=model, dataframe=df)
    trainer.train_model()
    trainer.save_model()
    trainer.evaluate_model()


if __name__ == "__main__":
    app()
