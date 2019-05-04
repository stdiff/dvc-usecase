"""
For a baseline model we do nothing
"""

from configparser import ConfigParser
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import DataFrameMetric

## load configuration
config = ConfigParser()
config.read("config.ini")

raw_path = Path(config["load"]["raw"])
training_set_path = Path(config["process"]["training"])
test_set_path = Path(config["process"]["test"])
metric_path = config["process"]["metric"]

@click.command()
@click.option("--test_size", default=0.3, type=float, help="proportion of the test set")
@click.option("--random_state", default=42, type=int, help="random seed for numpy")
def main(test_size:float, random_state:int):
    test_size = float(test_size)
    random_state = int(random_state)

    df = pd.read_csv(raw_path)
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    with DataFrameMetric(metric_path, name="splitting") as metric:
        metric.add_data(train=df_train)
        metric.add_data(test=df_test)

    df_train.to_csv(training_set_path, index_label=False)
    df_test.to_csv(test_set_path, index_label=False)

if __name__ == "__main__":
    main()
