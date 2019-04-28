"""
Simulate ingest application. This script fetch "california housing data set",
samples some rows from it and add the sampled rows in the data file (CSV).

Since this is just a prototype, we do not implement any complicated logic
(such as stacking data set without duplicate).
"""

import click
from pathlib import Path
from configparser import ConfigParser

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

config = ConfigParser()
config.read("config.ini")

csv_path = Path(config["data"]["raw"])
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
    scores = np.random.uniform(low=0, high=1, size=df.shape[0])
    selected = scores < sampling_rate

    return df.iloc[selected,:].copy()


@click.command()
@click.option("--round", default=1, type=int)
def fetch_data(round:int=1):
    sampling_rate = 0.2*round
    df_sampled = sampling(target_name=target_name, sampling_rate=sampling_rate, seed=seed)

    df_sampled.to_csv(str(csv_path), index_label=False)
    print("New data: %s" % csv_path)
    print("# rows  : %s" % df_sampled.shape[0])

if __name__ == "__main__":
    fetch_data()
