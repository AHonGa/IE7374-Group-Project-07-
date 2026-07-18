"""
Shared utility functions used across the pipeline.
"""

import yaml
import os
import pandas as pd


def load_config(config_path="configs/model_config.yaml"):
    """Load the YAML config file and return it as a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_dataframe(df, path, index=False):
    """Save a dataframe to CSV, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    print(f"Saved {len(df)} rows to {path}")


def print_section(title):
    """Print a visible section header for pipeline output."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
