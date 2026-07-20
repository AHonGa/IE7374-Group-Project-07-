"""
ROUGE evaluation utilities for generated adverse-event summaries.

A reference summary written by a human is required for each
generated summary.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from rouge_score import rouge_scorer


ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def calculate_rouge(
    reference: str,
    prediction: str,
) -> Dict[str, float]:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Parameters
    ----------
    reference:
        Human-written reference summary.
    prediction:
        Model-generated summary.

    Returns
    -------
    dict
        ROUGE F1 scores.
    """
    scorer = rouge_scorer.RougeScorer(
        ROUGE_TYPES,
        use_stemmer=True,
    )

    scores = scorer.score(
        str(reference),
        str(prediction),
    )

    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def evaluate_dataframe(
    df: pd.DataFrame,
    reference_column: str = "reference_summary",
    prediction_column: str = "bart_summary",
) -> pd.DataFrame:
    """
    Calculate ROUGE scores for each row in a DataFrame.

    Parameters
    ----------
    df:
        DataFrame containing reference and generated summaries.
    reference_column:
        Column containing human-written summaries.
    prediction_column:
        Column containing model-generated summaries.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with ROUGE columns.
    """
    required_columns = [
        reference_column,
        prediction_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    result_df = df.copy()

    rouge_results: List[Dict[str, float]] = []

    for _, row in result_df.iterrows():
        scores = calculate_rouge(
            reference=row[reference_column],
            prediction=row[prediction_column],
        )

        rouge_results.append(scores)

    rouge_df = pd.DataFrame(rouge_results)

    result_df = pd.concat(
        [
            result_df.reset_index(drop=True),
            rouge_df,
        ],
        axis=1,
    )

    return result_df


def summarize_rouge_scores(
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate average ROUGE scores.

    Parameters
    ----------
    scored_df:
        DataFrame containing ROUGE score columns.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame containing average ROUGE scores.
    """
    score_columns = [
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
    ]

    missing_columns = [
        column
        for column in score_columns
        if column not in scored_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing ROUGE columns: {missing_columns}"
        )

    averages = (
        scored_df[score_columns]
        .mean()
        .round(4)
        .to_frame()
        .T
    )

    averages.insert(
        0,
        "samples_evaluated",
        len(scored_df),
    )

    return averages


def run_rouge_evaluation(
    input_csv: str,
    output_csv: str,
    reference_column: str = "reference_summary",
    prediction_column: str = "bart_summary",
) -> None:
    """
    Run ROUGE evaluation from a CSV file.

    Parameters
    ----------
    input_csv:
        CSV containing reference and generated summaries.
    output_csv:
        Destination for row-level ROUGE results.
    reference_column:
        Name of reference-summary column.
    prediction_column:
        Name of generated-summary column.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    scored_df = evaluate_dataframe(
        df=df,
        reference_column=reference_column,
        prediction_column=prediction_column,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored_df.to_csv(
        output_path,
        index=False,
    )

    print("Average ROUGE scores:")
    print(summarize_rouge_scores(scored_df))

    print(f"Saved detailed results to: {output_path}")


if __name__ == "__main__":
    run_rouge_evaluation(
        input_csv="outputs/bart_reference_evaluation.csv",
        output_csv="outputs/rouge_results.csv",
    )
