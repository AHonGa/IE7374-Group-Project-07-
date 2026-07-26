# AI-Powered Adverse Event Summarizer from Patient Reviews

**Team 7 — IE7374: Generative AI**

## Overview

This project builds an NLP pipeline to detect and summarize adverse events described in patient reviews. We focus on extracting symptom mentions and producing conservative, source-grounded summaries of adverse events using pretrained transformer models.

## Models

- Summarization: BART (`facebook/bart-large-cnn`) for abstractive, source-grounded summaries.
- Adverse event classification: DistilBERT (`distilbert-base-uncased`) for efficient classification.

## Dataset

- UCI Drug Reviews Dataset (Kaggle: jessicali9530/kuc-hackathon-winter-2018), scraped from drugs.com.
- ~215k reviews (CSV with columns: uniqueID, drugName, condition, review, rating, date, usefulCount).
- Project will focus on Depression and Anxiety subsets (~15k reviews).

## Research Questions

1. How accurately can fine-tuned DistilBERT identify adverse event mentions versus a keyword baseline (precision/recall/F1)?
2. What symptom clusters appear most often in low-rated reviews (≤4) for antidepressants and anxiolytics, and do they match known side effect profiles?
3. Can BART generate adverse event summaries grounded in source reviews without making causal claims, and which prompt designs enforce that constraint?
4. How well does VADER sentiment scoring correlate with patient ratings, and where do the signals diverge?

## Plan of Action & Timeline

- Literature review & benchmarking (July 6–12): evaluate VADER, DistilBERT, BART; run small feasibility tests.
- Data pipeline & repo setup (July 6–18): merge CSVs, decode HTML entities, produce `review_clean` and `review_processed`, split focused subset (70/15/15).
- Model implementation & evaluation (July 13–25): fine-tune DistilBERT, run spaCy NER, cluster symptoms, apply BART to flagged reviews.
- Deliverables: Milestone 3 (July 19) — data pipeline and repo; Milestone 4 (July 26) — end-to-end pipeline; Milestone 5 (Aug 9) — technical report and presentation.

Frameworks: Python 3.10, HuggingFace Transformers, PyTorch, spaCy, NLTK, scikit-learn.

## Team Contributions

- Jin-woo Hong — Data Engineer: dataset sourcing, repo init, initial EDA. Deliverables: `drugsComTrain_raw.csv`, `drugsComTest_raw.csv`, `project_team_7.ipynb`.
- Yosephine Tong — Data Analyst: cleaning, preprocessing, EDA, VADER scoring, topic modeling. Deliverables: `df_clean.csv`, `df_focused.csv`, EDA notebook.
- Umang Khamar — NLP Engineer: DistilBERT fine-tuning, spaCy NER, symptom clustering, metrics (precision/recall/F1).
- Min Chang — AI Summarization Lead: BART pipeline, prompt design, ROUGE evaluation, Streamlit demo, report and slides.

Note: `review_clean` (minimally cleaned natural text) will be used for transformer models; `review_processed` is for classical NLP.

## Evaluation

- Classification: precision, recall, F1 against labeled adverse-event flags.
- Summaries: ROUGE scores and qualitative review; ensure conservative, source-grounded outputs.

### BART, T5, and GPT-2 Benchmarking Results

### Objective

This experiment compared BART, T5, and GPT-2 for summarizing patient drug reviews related to Depression and Anxiety. All three models were evaluated on the same fixed sample of 50 reviews.

The comparison focused on:

- Inference speed
- Output compression
- Coherence
- Factual grounding
- Adverse-event focus
- Overall summary quality

### Models

- BART: `facebook/bart-large-cnn`
- T5: `google-t5/t5-small`
- GPT-2: `gpt2`

### Dataset Sample

The benchmark used 50 reviews from `df_focused.csv`:

- 25 Depression reviews
- 25 Anxiety reviews
- Reviews with low ratings or negative VADER sentiment were prioritized
- Reviews contained at least 20 words
- The same sample was used for every model

### Quantitative Results

| Model | Total Time | Average Time per Review | Average Output-to-Source Ratio | Exact Copy Rate | Near-Copy Rate |
|---|---:|---:|---:|---:|---:|
| BART | 43.50 seconds | 0.870 seconds | 0.556 | 0.14 | 0.20 |
| T5 | 37.70 seconds | 0.754 seconds | 0.348 | 0.02 | 0.04 |
| GPT-2 | 53.17 seconds | 1.063 seconds | 0.814 | 0.00 | 0.00 |

T5 was the fastest model and produced the shortest outputs. GPT-2 was the slowest and produced outputs closest in length to the original reviews.

### Qualitative Evaluation

Ten reviews were selected for manual evaluation. Each model output was scored from 1 to 5 for coherence, factual grounding, adverse-event focus, and overall quality.

| Model | Coherence | Factual Grounding | Adverse-Event Focus | Overall Quality |
|---|---:|---:|---:|---:|
| BART | 4.50 | 5.00 | 3.30 | 3.40 |
| T5 | 3.50 | 4.90 | 3.60 | 3.10 |
| GPT-2 | 3.40 | 1.00 | 1.00 | 1.00 |

## Key Findings

### BART

BART produced the most coherent and factually grounded summaries. It generally preserved the meaning of the original reviews without introducing unsupported medical information.

However, BART occasionally copied short reviews nearly verbatim. It also sometimes preserved emotionally charged patient language rather than converting it into a neutral adverse-event summary.

A conservative natural-language prompt was tested, but BART summarized the instructions instead of following them. This is likely because `facebook/bart-large-cnn` is not instruction-tuned.

### T5

T5 was faster and generated more compressed summaries. It sometimes focused more directly on adverse effects than BART.

However, T5 outputs were often fragmented, grammatically awkward, or truncated. Some outputs also distorted sentence structure, although factual grounding remained substantially better than GPT-2.

### GPT-2

GPT-2 frequently generated unsupported medical claims. Examples included invented diagnoses such as cancer, brain tumors, and psychiatric conditions that were not present in the source reviews.

GPT-2 also reversed the meaning of some negative reviews and failed to consistently summarize adverse-event information.

### Final Model Selection

BART was selected for the final summarization pipeline.

Although T5 was faster and slightly more concise, BART achieved the strongest combination of coherence and factual grounding. These characteristics are especially important for patient-generated health text, where unsupported claims could be misleading.

GPT-2 was rejected because of frequent hallucinations and poor factual reliability.

## Milestone 4: Integrated BART Pipeline

For Milestone 4, the selected BART model was connected to the DistilBERT adverse-event classifier output.

The merged classifier dataset contained 11,814 reviews, including:

- 5,490 reviews with `ae_flag = 1`
- 6,324 reviews with `ae_flag = 0`
- No missing `review_clean` values
- No duplicate `uniqueID` values

Only reviews classified with `ae_flag = 1` were eligible for BART summarization.

A fixed sample of 10 classifier-flagged reviews was selected for the initial integrated pipeline test. Symptom extraction and clustering metadata were preserved when available.

### Integrated Pipeline

The implemented workflow first uses DistilBERT to classify reviews for adverse-event content. Reviews with `ae_flag = 1` are then selected for summarization. Available NER entities and symptom-cluster metadata are preserved, and BART generates a summary for each flagged review. The generated summaries are then compared with human-written reference summaries using ROUGE-1, ROUGE-2, and ROUGE-L.

## How to Run

### 1. Clone the Repository

Clone the project repository and move into the project folder:

```bash
git clone https://github.com/<repository-owner>/IE7374-Group-Project-07-.git
cd IE7374-Group-Project-07-
```

Replace `<repository-owner>` with the GitHub username or organization that owns the repository.

### 2. Install the Required Packages

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The main packages used by the summarization pipeline include:

- `torch`
- `transformers`
- `pandas`
- `rouge-score`
- `sentencepiece`
- `accelerate`

A CUDA-enabled GPU is recommended for faster BART inference, but the code can also run on CPU.

### 3. Prepare the Input Data

The integrated summarization pipeline uses the classifier output stored at:

```text
outputs/df_adverse_events.csv
```

The minimum required columns are:

- `uniqueID`
- `drugName`
- `condition`
- `review_clean`
- `rating`
- `ae_flag`

The pipeline also preserves the following fields when available:

- `ae_label`
- `ae_probabilities`
- `symptoms_extracted`
- `symptom_clusters`
- `cluster_id`

Only reviews with `ae_flag = 1` are selected for BART summarization.

### 4. Run the BART Summarization Script

From the repository root, run:

```bash
python src/summarizer.py
```

The reusable BART implementation is located at:

```text
src/summarizer.py
```

The model checkpoint used is:

```text
facebook/bart-large-cnn
```

The checkpoint is downloaded automatically from Hugging Face when the script runs and should not be committed to the repository.

Generated summaries are saved under:

```text
outputs/
```

### 5. Run the Full Colab Notebook

The complete benchmarking and integrated pipeline workflow is available in:

```text
notebooks/BART_Summarization.ipynb
```

To reproduce the notebook results:

1. Open the notebook in Google Colab.
2. Select `Runtime`.
3. Select `Change runtime type`.
4. Choose `T4 GPU`.
5. Run the notebook cells in order.
6. Upload `outputs/df_adverse_events.csv` when prompted.

The notebook performs:

- dataset loading and validation
- inspection of classifier and symptom fields
- filtering of reviews with `ae_flag = 1`
- fixed 10-review sample selection
- BART model loading
- summary generation
- inference-time measurement
- compression-ratio calculation
- human reference-summary preparation
- ROUGE-1, ROUGE-2, and ROUGE-L evaluation
- final output export

### 6. Run ROUGE Evaluation

The reusable ROUGE evaluation script is located at:

```text
utils/rouge_evaluation.py
```

The evaluation input must contain the following columns:

- `reference_summary`
- `bart_summary`

Run the evaluation from the repository root:

```bash
python utils/rouge_evaluation.py
```

The script calculates:

- ROUGE-1 precision, recall, and F1
- ROUGE-2 precision, recall, and F1
- ROUGE-L precision, recall, and F1
- average scores across all evaluated summaries

The completed Milestone 4 ROUGE outputs are stored at:

```text
outputs/bart_rouge_detailed_results.csv
outputs/bart_rouge_summary.csv
```

### 7. Review the Milestone 4 Outputs

The main integrated summarization and evaluation files are:

```text
outputs/bart_integrated_summaries_10.csv
outputs/bart_reference_evaluation_completed.csv
outputs/bart_rouge_detailed_results.csv
outputs/bart_rouge_summary.csv
```

These files contain the classifier-flagged reviews, BART-generated summaries, human-written reference summaries, detailed row-level ROUGE scores, and average ROUGE results.

## References

- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text.
- Lewis, M., et al. (2020). BART: Denoising sequence-to-sequence pre-training for NLG.
- Sanh, V., et al. (2019). DistilBERT: smaller, faster BERT.
- Sarker, A., & Gonzalez, G. (2015). ADR detection from social media.
- Kaggle dataset: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
