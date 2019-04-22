"""
Simulate ingest application. This script fetch "california housing data set",
samples some rows from it and add the sampled rows in the data file (CSV).

Since this is just a prototype, we do not implement any complicated logic
(such as stacking data set without duplicate).
"""

from pathlib import Path
from configparser import ConfigParser

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

config = ConfigParser()
config.read("../config.ini")

csv_path = Path("..").joinpath(config["data"]["raw"])
data_dir =  csv_path.parent
target_name = config["data"]["target"]

sampling_rate = 0.2
seed = 51

def sampling(target_name:str="MedHouseVal", sampling_rate:float=0.2,
             seed:int=None) -> pd.DataFrame:
    """
    return sampled data from California Housing data set

    :param target_name: the name of the target variable
    :param sampling_rate: rate of sampling (against the whole)
    :param seed: random seed for numpy
    :return: DataFrame of sampled instances
    """
    if seed is not None:
        np.random.seed(seed)

    data = fetch_california_housing(download_if_missing=True)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[target_name] = data.target

    ## sampling
    chunk_size = int(sampling_rate*df.shape[0])
    sampled_rows = np.random.randint(df.shape[0], size=chunk_size)

    return df.iloc[sampled_rows,:].copy()

if __name__ == "__main__":

    if csv_path.exists():
        ## data exists => shifting the random seed => sampling data
        df_stacked = pd.read_csv(str(csv_path))
        print("You have already %s rows" % df_stacked.shape[0])

        df_sampled = sampling(target_name=target_name, sampling_rate=sampling_rate,
                              seed=seed + df_stacked.shape[0])

        df_stacked = pd.concat([df_stacked, df_sampled], axis=0)

    else:
        ## no data exists => just sampling data
        df_stacked = sampling(target_name=target_name, sampling_rate=sampling_rate, seed=seed)
        print("You have no data yet.")

    df_stacked.to_csv(str(csv_path), index_label=False)
    print("New rows are added to %s" % csv_path)
