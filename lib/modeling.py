"""
Functions for training a mathematical model.
"""

from typing import Tuple, Union, Dict, List, Any

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

def training_a_model(training_data:pd.DataFrame, target_name:str, cv:int=5,
                     **kwargs) -> Tuple[GridSearchCV,str]:
    """
    The main function of the training.

    :param training_data: Data Frame (training data)
    :param target_name: neme of the target variable
    :param cv: number of folds in cross-validation
    :param kwargs: other arguments
    :return: (fitted GridSearchCV instance, model name)
    """

    """
    You need to create the following three variables before training.
    
    pipeline: BaseEstimator or Pipeline instance
    param_grid: grid of hyperparameter
    name      
    """

    pipeline, param_grid, name = elastic_net(**kwargs)
    #pipeline, param_grid, name = random_forest(**kwargs)

    ### training
    model = GridSearchCV(pipeline, param_grid, cv=cv, scoring="neg_mean_squared_error")
    model.fit(training_data.drop(target_name,axis=1), training_data[target_name])
    return model, name


def elastic_net(random_state:int=51) \
        -> Tuple[Union[BaseEstimator,Pipeline], Dict[str,List[Any]], str]:
    """
    A simple baseline model

    :param random_state: random state
    :return: (unfitted) Pipeline model, param_grid, "baseline model"
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import ElasticNet

    pipeline = Pipeline([("std_scaler", StandardScaler()),
                         ("enet", ElasticNet(random_state=random_state))])

    param_grid = {"enet__alpha": [0.01, 0.1, 1],
                  "enet__l1_ratio": [0.1, 0.4, 0.7, 1]}

    return pipeline, param_grid, "baseline model"


def random_forest(random_state:int=51) \
        -> Tuple[Union[BaseEstimator,Pipeline], Dict[str,List[Any]], str]:
    """
    A plain random forest model

    :param random_state: random state
    :return: (unfitted) Pipeline model, param_grid, "random forest"
    """
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(random_state=random_state)
    param_grid = {"max_depth": [2, 5, 7],
                  "n_estimators": [10, 30, 100]}

    return model, param_grid, "random forest"