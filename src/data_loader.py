"""
Data loading module for the adverse event summarizer pipeline.

Loads and merges the drugsComTrain_raw.csv and drugsComTest_raw.csv files
from data/processed/, adds a split label, and returns a single dataframe
for downstream preprocessing.

Role 2 deliverable — Yosephine Tong
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, print_section


def load_raw_data(train_path=None, test_path=None, config_path="configs/model_config.yaml"):
    """
    Load train and test CSVs, tag each row with its split origin,
    and return a single merged dataframe.

    Parameters
    ----------
    train_path : str, optional
        Path to the training CSV. Defaults to config value.
    test_path : str, optional
        Path to the test CSV. Defaults to config value.
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with a 'split' column ('train' or 'test').
    """
    config = load_config(config_path)

    train_path = train_path or config["data"]["train_path"]
    test_path  = test_path  or config["data"]["test_path"]

    print_section("Loading raw data")

    df_train = pd.read_csv(train_path, encoding="utf-8")
    df_test  = pd.read_csv(test_path,  encoding="utf-8")

    df_train["split"] = "train"
    df_test["split"]  = "test"

    df = pd.concat([df_train, df_test], ignore_index=True)

    print(f"  Train rows : {len(df_train):,}")
    print(f"  Test rows  : {len(df_test):,}")
    print(f"  Total rows : {len(df):,}")
    print(f"  Columns    : {list(df.columns)}")

    return df


def get_focused_subset(df, conditions=None, config_path="configs/model_config.yaml"):
    """
    Filter the dataframe to a specific set of medical conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Full merged dataframe.
    conditions : list of str, optional
        Conditions to keep. Defaults to config value (Depression, Anxiety).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only the specified conditions.
    """
    config = load_config(config_path)
    conditions = conditions or config["data"]["focus_conditions"]

    df_focused = df[df["condition"].isin(conditions)].copy()

    print_section(f"Focused subset: {conditions}")
    print(f"  Rows kept : {len(df_focused):,} of {len(df):,}")
    for cond in conditions:
        n = (df_focused["condition"] == cond).sum()
        print(f"    {cond}: {n:,} reviews")

    return df_focused


if __name__ == "__main__":
    df = load_raw_data()
    df_focused = get_focused_subset(df)
    print(df_focused.head(3))
