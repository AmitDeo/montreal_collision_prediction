import os

import pandas as pd


def load_data(filename: str) -> pd.DataFrame:
    """
    Load data from path
    """
    data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", filename))

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} does not exist")

    df = pd.read_csv(data_file)
    return df
