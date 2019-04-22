"""
An elastic net as a baseline
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from configparser import ConfigParser
from pathlib import Path

import pandas as pd
import joblib

sys.path.append("03-modeling")
from utils import Metric

## load configuration
config = ConfigParser()
config.read("config.ini")

tz = config["general"]["tz"]
training_data = Path(config["data"]["training"])
test_data = Path(config["data"]["test"])
target_name = config["data"]["target"]
model_path = Path(config["model"]["model"])
metric_path = Path(config["model"]["metric"])


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error

    df_train = pd.read_csv(training_data)

    ## train a model
    pipeline = Pipeline([("std_scaler", StandardScaler()),
                         ("enet", ElasticNet(random_state=3))])

    param_grid = {"enet__alpha": [0.01, 0.1, 1],
                  "enet__l1_ratio": [0.1, 0.4, 0.7, 1]}

    model = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error")

    model.fit(df_train.drop(target_name, axis=1), df_train[target_name])

    ## write metric
    metric = Metric(str(metric_path), tz=tz)
    metric.read_off_cv_results_(model)

    yhat_train = model.predict(df_train.drop(target_name, axis=1))
    metric.rmse_train = mean_squared_error(df_train[target_name], yhat_train)

    df_test = pd.read_csv(test_data)
    yhat_test = model.predict(df_test.drop(target_name, axis=1))
    metric.rmse_test = mean_squared_error(df_test[target_name], yhat_test)

    metric.save_metrics()
    joblib.dump(model.best_estimator_, model_path)