"""
The logic for the data processing. Only `to_feature_matrix` is called
from the entry point.
"""

from typing import Tuple
import pandas as pd

def to_feature_matrix(**kwargs) -> Tuple[pd.DataFrame,str]:
    """
    This function is the main function which is called from the entry point
    of data processing.

    :return: (DataFrame (feature matrix), name of the logic)
    """
    df = pd.read_csv(kwargs["path"])

    return identity_map(df)


def identity_map(df:pd.DataFrame) -> Tuple[pd.DataFrame,str]:
    """
    Does nothing.

    :param df: an arbitrary DataFrame
    :return: (the same DataFrame, "nothing")
    """
    return df, "nothing"


