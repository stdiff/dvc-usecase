"""
Definition of Metric class (For RMSE)
"""

import os
from datetime import datetime
from pytz import timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

class Metric:
    def __init__(self, metric_path:str, tz:str="Europe/Berlin"):
        self.metric_path = Path(metric_path)
        self.user = os.getenv("USER")
        self.tz = timezone(tz)
        self._rmse_test = None
        self._rmse_train = None
        self._mean_rmse_validation = None
        self.std_rmse_validation = None
        self._model = None

        self.keys = ["user", "timestamp", "rmse_test", "rmse_train",
                     "mean_rmse_validation", "std_rmse_validation",
                     "model"]

    @property
    def rmse_test(self) -> float:
        return self._rmse_test

    @rmse_test.setter
    def rmse_test(self, rmse:float):
        if rmse < 0:
            raise AttributeError("RMSE must be non-negative. You gave %s" % rmse)
        self._rmse_test = rmse

    @property
    def rmse_train(self) -> float:
        return self._rmse_train

    @rmse_train.setter
    def rmse_train(self, rmse:float):
        if rmse < 0:
            raise AttributeError("RMSE must be non-negative. You gave %s" % rmse)
        self._rmse_train = rmse

    @property
    def mean_rmse_validation(self) -> float:
        return self._mean_rmse_validation

    @mean_rmse_validation.setter
    def mean_rmse_validation(self, rmse:float):
        if rmse < 0:
            raise AttributeError("RMSE must be non-negative. You gave %s" % rmse)
        self._mean_rmse_validation = rmse

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, description:str):
        if not isinstance(description, str):
            msg = "The model description must be string. You gave %s" % type(description)
            raise AttributeError(msg)
        self._model = description


    def read_off_cv_results_(self, model:GridSearchCV):
        best_score = pd.DataFrame(model.cv_results_) \
                       .sort_values(by="mean_test_score", ascending=False)\
                       .iloc[0,:]
        self.mean_rmse_validation = -best_score["mean_test_score"]
        self.std_rmse_validation = best_score["std_test_score"]
        self.model = str(model.best_estimator_)


    def save_metrics(self):
        for key in self.keys[2:]:
            if getattr(self, key) is None:
                raise ValueError("No value for %s is given" % key)

        self.timestamp = datetime.now(tz=self.tz).isoformat(sep=" ")

        with self.metric_path.open("w") as fo:
            for key in self.keys:
                val = getattr(self, key)
                fo.write("%s: %s\n" % (key,val))


if __name__ == "__main__":
    metric = Metric("./test.txt")
    metric.rmse_test = 0.3
    metric.rmse_train = 0.2
    metric.mean_rmse_validation = 0.25
    metric.std_rmse_validation = 0.04
    metric.model = "hoge"

    metric.save_metrics()
