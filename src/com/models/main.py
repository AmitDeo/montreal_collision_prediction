import typer

from com.models.trainer import Trainer
from com.models.utils.load_data import load_data
from com.models.utils.select_model import select_model

app = typer.Typer()


@app.command()
def train(model_name: str, data: str = "data.csv"):
    """
    The function to train the model as defined in model
    """

    model = select_model(model_name)

    df = load_data(data)
    trainer = Trainer(model=model, dataframe=df)
    trainer.train_model()
    trainer.evaluate_model()
    trainer.save_model()


@app.command()
def evaluate(model_name: str, data: str = "data.csv"):
    """
    The function to train the model as defined in model
    """
    model = select_model(model_name)

    df = load_data(data)

    trainer = Trainer(model=model, dataframe=df)
    trainer.load_model()
    trainer.evaluate_model()


if __name__ == "__main__":
    app()
