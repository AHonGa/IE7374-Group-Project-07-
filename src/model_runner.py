"""
End-to-end pipeline runner for the adverse event summarizer.

Usage:
    python src/model_runner.py

Steps:
    1. Load and preprocess the dataset (Role 2 — Yosephine Tong)
    2. Run adverse event classification with DistilBERT (Role 3 — Umang Khamar)
    3. Run BART summarization on flagged reviews (Role 4 — Min Chang)
    4. Save generated summaries to outputs/samples.txt

Note: Steps 2 and 3 are placeholders pending Role 3 and Role 4 implementation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader    import load_raw_data, get_focused_subset
from src.preprocessing  import preprocess_dataframe, split_focused_subset
from utils.helpers      import load_config, save_dataframe, print_section


def run_pipeline(config_path="configs/model_config.yaml"):
    config = load_config(config_path)

    # ── Step 1: Data loading and preprocessing (Role 2) ──────────────────────
    df         = load_raw_data(config_path=config_path)
    df_clean   = preprocess_dataframe(df)
    df_focused = get_focused_subset(df_clean, config_path=config_path)

    save_dataframe(df_clean,   config["data"]["clean_output"])
    save_dataframe(df_focused, config["data"]["focused_output"])

    df_train, df_val, df_test = split_focused_subset(df_focused, config_path=config_path)

    # ── Step 2: Adverse event classification (Role 3 — placeholder) ──────────
    print_section("Adverse Event Classification (Role 3 — pending)")
    print("  DistilBERT classification not yet implemented.")
    print("  Expected output: df_focused with 'adverse_event' flag column.")

    # ── Step 3: BART summarization (Role 4 — placeholder) ────────────────────
    print_section("BART Summarization (Role 4 — pending)")
    print("  BART summarization not yet implemented.")
    print("  Expected output: outputs/samples.txt with generated summaries.")

    print_section("Pipeline run complete")
    print(f"  Cleaned dataset : {config['data']['clean_output']}")
    print(f"  Focused subset  : {config['data']['focused_output']}")


if __name__ == "__main__":
    run_pipeline()
