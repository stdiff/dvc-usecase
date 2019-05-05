"""
The entry point of training a mathematical model. This script calls
"lib.modeling.training_a_model". The model training must be implemented
in the function.
"""

import warnings
warnings.filterwarnings("ignore")

from configparser import ConfigParser
from pathlib import Path
import click

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

from lib import RMSEMetric
from lib.modeling import training_a_model

## load configuration
config = ConfigParser()
config.read("config.ini")

tz = config["general"]["tz"]
training_data = Path(config["processing"]["training"])
test_data = Path(config["processing"]["test"])
target_name = config["general"]["target"]
model_path = Path(config["modeling"]["model"])
metric_path = Path(config["modeling"]["metric"])

@click.command()
@click.option("--random_state", default=42, type=int, help="random seed for numpy")
def main(random_state:int=42):
    """
    Train a model and store the fitted model.

    :param random_state: random state
    """

    df_train = pd.read_csv(training_data)
    model, name = training_a_model(df_train, target_name,
                                   random_state=random_state)

    with RMSEMetric(str(metric_path), tz=tz, name=name) as metric:
        metric.read_off_cv_results_(model)

        yhat_train = model.predict(df_train.drop(target_name, axis=1))
        metric.rmse_train = mean_squared_error(df_train[target_name], yhat_train)

        df_test = pd.read_csv(test_data)
        yhat_test = model.predict(df_test.drop(target_name, axis=1))
        metric.rmse_test = mean_squared_error(df_test[target_name], yhat_test)

    joblib.dump(model.best_estimator_, model_path)

if __name__ == "__main__":
    main()