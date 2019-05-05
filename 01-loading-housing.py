"""
Simulate ingest application. This script fetch "california housing data set",
samples some rows from it and add the sampled rows in the data file (CSV).

Since this is just a prototype, we do not implement any complicated logic
(such as stacking data set without duplicate).
"""

"""
We should have a code for each data source. Because 
- the frequency of retrieval depends on the data source. 
- the retrieval method also depends on the data source.  
"""

import click
from pathlib import Path
from configparser import ConfigParser

import pandas as pd

from lib import DataFrameMetric

config = ConfigParser()
config.read("config.ini")

csv_path = Path(config["loading_housing"]["raw_data"])
metric_path = Path(config["loading_housing"]["metric"])
target_name = config["general"]["target"]
tz = config["general"]["tz"]

def fetch_data(round:int) -> pd.DataFrame:
    """
    Retrieve the data you need.

    :param round: round number
    :return: DataFrame
    """

    """
    You have to implement this function. This function can take 
    an option (from the command line) and produces a DataFrame.
    (If you need a different kind of output, you also need to 
    modify the main function.     
    """
    import numpy as np
    from sklearn.datasets import fetch_california_housing

    sampling_rate = 0.2*round
    np.random.seed(51)

    data = fetch_california_housing(download_if_missing=True)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[target_name] = data.target

    ## sampling
    scores = np.random.uniform(low=0, high=1, size=df.shape[0])
    selected = scores < sampling_rate

    return df.iloc[selected,:].copy()


@click.command()
@click.option("--round", default=1, type=int)
def main(round:int):
    df = fetch_data(round)

    with DataFrameMetric(metric_path, name="housing", tz=tz) as metric:
        metric.add_data(housing=df)

    """
    Store the retrieved data

    Depending on the data, you might want to modify the following
    lines. 

    For example you can retrieve the (relatively) same data at any time,
    you might want to overwrite the file. If the data largely depends on
    when you retrieve the data, then you might want to have one file
    for each retrieval. (In this case you need to also be careful about 
    the value of -o option. This should be no directory.) 
    """

    df.to_csv(str(csv_path), index_label=False)


if __name__ == "__main__":
    main()