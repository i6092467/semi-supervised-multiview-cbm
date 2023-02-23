"""
Utility functions for logging
"""
import os
from os.path import join


def save_data_split(splits_dir, trainset, train_ids, validset, val_ids):
    """
    Write training and validation sample names to a file for each fold.
    """

    if os.path.exists(join(splits_dir, "data_split.csv")):
        os.remove(join(splits_dir, "data_split.csv"))

    with open(join(splits_dir, "data_split.csv"), "w") as f:
        f.write("Train samples\n")
        for idx in train_ids:
            img_code = list(trainset.labels)[idx]
            f.write(img_code)
            f.write("\n")
        f.write("\nValidation samples\n")
        for idx in val_ids:
            img_code = list(validset.labels)[idx]
            f.write(img_code)
            f.write("\n")
