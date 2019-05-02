import unittest

import shutil
from tempfile import mkdtemp
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris, load_boston
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils import DataFrameMetric, RMSEMetric

def iris_dataframe() -> pd.DataFrame:
    data_iris = load_iris()
    df = pd.DataFrame(data_iris.data, columns=["x%s" % i for i in range(4)])
    df["label"] = [data_iris.target_names[i] for i in data_iris.target]
    return df

def boston_dataframe() -> pd.DataFrame:
    data = load_boston()
    df = pd.DataFrame(data.data, columns=["x%02d" % i for i in range(13)])
    df["val"] = data.target
    return df


class GoToTempDir:
    def __init__(self):
        self.tmp_dir = mkdtemp()

    def __enter__(self) -> Path:
        return Path(self.tmp_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.tmp_dir)


class MetricTest(unittest.TestCase):
    def test_DataFrameMetric(self):
        df = iris_dataframe()
        df_train = df.iloc[:91,:]
        df_test = df.iloc[91:,:]

        with GoToTempDir() as tmp_dir:
            metric_path = tmp_dir.joinpath("metrics.txt")
            metric = DataFrameMetric(str(metric_path), name="iris")
            metric.add_data(train=df_train)
            metric.add_data(test=df_test)
            metric.save_metrics()

            keys = ["user","timestamp","name","train_nrow","train_ncol",
                    "test_nrow", "test_ncol"]

            k2v = dict()
            with metric_path.open("r") as fo:
                for line in fo:
                    row = line.rstrip().split(": ")
                    self.assertTrue(len(row) >= 2)
                    self.assertIn(row[0],keys)
                    k2v[row[0]] = row[1]

            self.assertEqual(int(k2v["train_nrow"]), 91)
            self.assertEqual(int(k2v["train_ncol"]), 5)
            self.assertEqual(int(k2v["test_nrow"]), 150 - 91)
            self.assertEqual(int(k2v["test_ncol"]), 5)


    def test_RMSEMetric(self):
        df = boston_dataframe()
        df_train, df_test = train_test_split(df, test_size=0.4, random_state=1)

        param_grid = {"alpha": [0.1, 1], "l1_ratio": [0.1, 0.5, 0.7]}

        model = GridSearchCV(ElasticNet(random_state=3), param_grid,
                             scoring="neg_mean_squared_error", cv=3,
                             return_train_score=True)
        model.fit(df_train.drop("val", axis=1), df_train["val"])

        yhat = model.predict(df_train.drop("val", axis=1))
        train_loss = mean_squared_error(df_train["val"], yhat)

        yhat = model.predict(df_test.drop("val", axis=1))
        test_loss = mean_squared_error(df_test["val"], yhat)

        with GoToTempDir() as tmp_dir:
            metric_path = tmp_dir.joinpath("metric.txt")
            metric = RMSEMetric(str(metric_path))

            ## rmse must be non-negative
            with self.assertRaises(AttributeError):
                metric.rmse_train = -1.1

            with self.assertRaises(AttributeError):
                metric.rmse_test = -0.00000001

            with self.assertRaises(AttributeError):
                metric.mean_rmse_validation = -4

            metric.rmse_train = train_loss
            metric.read_off_cv_results_(model)

            with self.assertRaises(ValueError):
                ## because the test score is not given, ValueError is risen.
                metric.save_metrics()

            metric.rmse_test = test_loss
            metric.save_metrics()

            metric_dict = dict()
            with metric_path.open("r") as fo:
                for line in fo:
                    vals = line.rstrip().split(": ")
                    self.assertTrue(len(vals) >= 2)
                    metric_dict[vals[0]] = vals[1]

            self.assertAlmostEqual(float(metric_dict["rmse_test"]),27.258223349178355)
            self.assertAlmostEqual(float(metric_dict["rmse_train"]), 21.555756107437798)
            self.assertEqual(metric_dict["model"], "ElasticNet")


    def test_RMSEMetric_pipeline(self):
        df = boston_dataframe()
        df_train, df_test = train_test_split(df, test_size=0.4, random_state=1)

        pipeline = Pipeline([("mms", MinMaxScaler()), ("enet", ElasticNet(random_state=3))])

        param_grid = {"enet__alpha": [0.1, 1] ,
                      "enet__l1_ratio": [0.1, 0.5, 0.7]}

        model = GridSearchCV(pipeline, param_grid, scoring="neg_mean_squared_error", cv=3,
                             return_train_score=True)
        model.fit(df_train.drop("val", axis=1), df_train["val"])

        yhat = model.predict(df_train.drop("val", axis=1))
        train_loss = mean_squared_error(df_train["val"], yhat)

        yhat = model.predict(df_test.drop("val", axis=1))
        test_loss = mean_squared_error(df_test["val"], yhat)

        with GoToTempDir() as tmp_dir:
            metric_path = tmp_dir.joinpath("metric.txt")
            metric = RMSEMetric(str(metric_path))
            metric.rmse_train = train_loss
            metric.read_off_cv_results_(model)
            metric.rmse_test = test_loss
            metric.save_metrics()

            metric_dict = dict()
            with metric_path.open("r") as fo:
                for line in fo:
                    vals = line.rstrip().split(": ")
                    self.assertTrue(len(vals) >= 2)
                    metric_dict[vals[0]] = vals[1]

            self.assertAlmostEqual(float(metric_dict["rmse_test"]),40.73907557425112)
            self.assertAlmostEqual(float(metric_dict["rmse_train"]), 29.51089563013877)
            self.assertEqual(metric_dict["model"], "MinMaxScaler|ElasticNet")


if __name__ == "__main__":
    unittest.main()