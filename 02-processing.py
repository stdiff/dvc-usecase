"""
The main entry point of the data processing. This script applies
"lib.processing.to_feature_matrix" to the raw data and stores
the result. Your logic must be implemented in lib/processing.py.
"""

from configparser import ConfigParser
from pathlib import Path

import click
from sklearn.model_selection import train_test_split

from lib import DataFrameMetric
from lib.processing import to_feature_matrix

## load configuration
config = ConfigParser()
config.read("config.ini")

raw_path = Path(config["loading_housing"]["raw_data"])
training_set_path = Path(config["processing"]["training"])
test_set_path = Path(config["processing"]["test"])
metric_path = config["processing"]["metric"]

@click.command()
@click.option("--test_size", default=0.3, type=float, help="proportion of the test set")
@click.option("--random_state", default=42, type=int, help="random seed for numpy")
def main(test_size:float, random_state:int):

    test_size = float(test_size)
    random_state = int(random_state)

    df, name = to_feature_matrix(path=raw_path)
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    with DataFrameMetric(metric_path, name=name) as metric:
        metric.add_data(train=df_train)
        metric.add_data(test=df_test)

    df_train.to_csv(training_set_path, index_label=False)
    df_test.to_csv(test_set_path, index_label=False)

if __name__ == "__main__":
    main()
