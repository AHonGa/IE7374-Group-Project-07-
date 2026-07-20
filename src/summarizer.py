"""
BART summarization pipeline for patient drug reviews.

This module loads facebook/bart-large-cnn and provides reusable
functions for summarizing individual reviews or an entire DataFrame.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL_NAME = "facebook/bart-large-cnn"


class BARTSummarizer:
    """Reusable BART summarization pipeline."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the BART tokenizer and model.

        Parameters
        ----------
        model_name:
            Hugging Face model checkpoint.
        device:
            Device to use. Defaults to CUDA when available,
            otherwise CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model_name = model_name

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model: {model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=(
                torch.float16
                if self.device.type == "cuda"
                else torch.float32
            ),
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"BART loaded on: {self.device}")

    def summarize(
        self,
        review: str,
        max_input_length: int = 1024,
        max_summary_length: int = 64,
        min_summary_length: int = 5,
        num_beams: int = 4,
    ) -> str:
        """
        Generate a summary for one patient review.

        Parameters
        ----------
        review:
            Clean patient review text.
        max_input_length:
            Maximum number of input tokens.
        max_summary_length:
            Maximum number of output tokens.
        min_summary_length:
            Minimum number of output tokens.
        num_beams:
            Beam-search width.

        Returns
        -------
        str
            Generated summary.
        """
        if not isinstance(review, str) or not review.strip():
            return ""

        inputs = self.tokenizer(
            review.strip(),
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_summary_length,
                min_length=min_summary_length,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
            )

        return self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
        ).strip()

    def summarize_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "review_clean",
        output_column: str = "bart_summary",
    ) -> pd.DataFrame:
        """
        Generate BART summaries for all rows in a DataFrame.

        Parameters
        ----------
        df:
            Input DataFrame.
        text_column:
            Column containing review text.
        output_column:
            Name of generated-summary column.

        Returns
        -------
        pd.DataFrame
            Copy of the input DataFrame with generated summaries.
        """
        if text_column not in df.columns:
            raise KeyError(
                f"Required column '{text_column}' was not found."
            )

        result_df = df.copy()

        result_df[output_column] = (
            result_df[text_column]
            .fillna("")
            .astype(str)
            .apply(self.summarize)
        )

        return result_df


def run_pipeline(
    input_csv: str,
    output_csv: str,
    text_column: str = "review_clean",
) -> None:
    """
    Load reviews from CSV, generate BART summaries, and save results.

    Parameters
    ----------
    input_csv:
        Path to input CSV.
    output_csv:
        Path to output CSV.
    text_column:
        Column containing review text.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    summarizer = BARTSummarizer()

    results_df = summarizer.summarize_dataframe(
        df=df,
        text_column=text_column,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved summaries to: {output_path}")


if __name__ == "__main__":
    run_pipeline(
        input_csv="data/processed/df_focused.csv",
        output_csv="outputs/bart_summaries.csv",
    )
