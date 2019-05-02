"""
Helper classes for DVC
"""

import os
from datetime import datetime
from pytz import timezone
from pathlib import Path
from collections import OrderedDict

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class Metric:
    def __init__(self, metric_path:str, tz:str="UTC"):
        """
        Parent class of helper classes for DVC

        :param metric_path: path to the metric file
        :param tz: timezone such as UTC
        """
        self.user = os.getenv("USER")
        self.metric_path = Path(metric_path)
        self.tz = timezone(tz)
        self.keys = ["user", "timestamp"]


    def save_metrics(self):
        for key in self.keys[2:]:
            if getattr(self, key) is None:
                raise ValueError("No value for %s is given" % key)

        self.timestamp = datetime.now(tz=self.tz).isoformat(sep=" ")

        with self.metric_path.open("w") as fo:
            for key in self.keys:
                val = getattr(self, key)
                fo.write("%s: %s\n" % (key, val))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_metrics()


class DataFrameMetric(Metric):
    def __init__(self, metric_path:str, name:str,
                 tz:str="UTC", **kwargs):
        """
        Helper class for DVC metric file for data sets.

        :param metric_path: path to the metric file
        :param name: name of the data
        :param tz: timezone such as UTC.
        :param kwargs: data_name=DataFrame
        """
        super().__init__(metric_path, tz)
        self.name = name
        self.keys.append("name")

        self.data = OrderedDict()
        self.add_data(**kwargs)


    def add_data(self,**kwargs):
        """
        Append data to the instance

        :param kwargs: data_name=DataFrame
        """
        for key, val in kwargs.items():
            if not isinstance(val,pd.DataFrame):
                raise ValueError("The given data must be pandas DataFrame.")

            key_nrow = "%s_nrow" % key
            self.keys.append(key_nrow)
            setattr(self, key_nrow, val.shape[0])

            key_ncol = "%s_ncol" % key
            self.keys.append(key_ncol)
            setattr(self, key_ncol, val.shape[1])


class RMSEMetric(Metric):
    def __init__(self, metric_path:str, tz:str="UTC"):
        """
        Metric object for regression (RMSE)

        :param metric_path: path to the metric file (str)
        :param tz: timezone such as "UTC".
        """
        super().__init__(metric_path, tz)
        self.keys.extend(["rmse_test",
                          "rmse_train",
                          "mean_rmse_validation",
                          "std_rmse_validation",
                          "model",
                          "best_params"])
        self._rmse_train = None
        self._rmse_test = None
        self._mean_rmse_validation = None
        self.std_rmse_validation = None
        self._model = None
        self.best_params = None


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
        """
        set model, best_params, mean_rmse_validation and std_rmse_validation
        by using a fitted GridSearchCV instance.

        :param model: fitted GridSearchCV instance.
        """

        if isinstance(model.best_estimator_, Pipeline):
            self.model = "|".join([type(pair[1]).__name__ for pair in model.best_estimator_.steps])
        else:
            self.model = type(model.best_estimator_).__name__

        self.best_params = model.best_params_

        best_score = pd.DataFrame(model.cv_results_) \
                       .sort_values(by="mean_test_score", ascending=False)\
                       .iloc[0,:]
        self.mean_rmse_validation = -best_score["mean_test_score"]
        self.std_rmse_validation = best_score["std_test_score"]
