"""
For a baseline model we do nothing
"""

from configparser import ConfigParser
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split

## load configuration
config = ConfigParser()
config.read("../config.ini")

raw_path = Path("..").joinpath(config["data"]["raw"])
training_set_path = Path("..").joinpath(config["data"]["training"])
test_set_path = Path("..").joinpath(config["data"]["test"])


@click.command()
@click.option("--test_size", default=0.3, type=float, help="proportion of the test set")
@click.option("--random_state", default=42, type=int, help="random seed for numpy")
def main(test_size:float, random_state:int):
    test_size = float(test_size)
    random_state = int(random_state)

    df = pd.read_csv(raw_path)
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train.to_csv(training_set_path, index_label=False)
    df_test.to_csv(test_set_path, index_label=False)
    print("saved")


if __name__ == "__main__":
    main()
