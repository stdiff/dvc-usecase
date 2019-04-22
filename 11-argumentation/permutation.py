"""
data argumentation: permutation of the RBG-dimension.
"""

from scipy.misc import imread, imsave
from itertools import permutations
from configparser import ConfigParser
from pathlib import Path

## load configuration
config = ConfigParser()
config.read("config.ini")
img_original_path = Path(config["image"]["img_original"])
dir_manipulated = Path(config["image"]["dir_manipulated"])

if __name__ == "__main__":
    img_original = imread(str(img_original_path))

    ## Since "dvc run" removes the target directory, we create it.
    if not dir_manipulated.exists():
        dir_manipulated.mkdir()

    for a,b,c in permutations(range(3),3):
        img_manipulated = img_original[:,:,[a,b,c]].copy()
        file_path = dir_manipulated.joinpath("image_%s%s%s.png" % (a,b,c))
        imsave(str(file_path), img_manipulated)
