# BART, T5, and GPT-2 Benchmarking Results

## Objective

This experiment compared BART, T5, and GPT-2 for summarizing patient drug reviews related to Depression and Anxiety. All three models were evaluated on the same fixed sample of 50 reviews.

The comparison focused on:

- Inference speed
- Output compression
- Coherence
- Factual grounding
- Adverse-event focus
- Overall summary quality

## Models

- BART: `facebook/bart-large-cnn`
- T5: `google-t5/t5-small`
- GPT-2: `gpt2`

## Dataset Sample

The benchmark used 50 reviews from `df_focused.csv`:

- 25 Depression reviews
- 25 Anxiety reviews
- Reviews with low ratings or negative VADER sentiment were prioritized
- Reviews contained at least 20 words
- The same sample was used for every model

## Quantitative Results

| Model | Total Time | Average Time per Review | Average Output-to-Source Ratio | Exact Copy Rate | Near-Copy Rate |
|---|---:|---:|---:|---:|---:|
| BART | 43.50 seconds | 0.870 seconds | 0.556 | 0.14 | 0.20 |
| T5 | 37.70 seconds | 0.754 seconds | 0.348 | 0.02 | 0.04 |
| GPT-2 | 53.17 seconds | 1.063 seconds | 0.814 | 0.00 | 0.00 |

T5 was the fastest model and produced the shortest outputs. GPT-2 was the slowest and produced outputs closest in length to the original reviews.

## Qualitative Evaluation

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

## Final Model Selection

BART was selected for the final summarization pipeline.

Although T5 was faster and slightly more concise, BART achieved the strongest combination of coherence and factual grounding. These characteristics are especially important for patient-generated health text, where unsupported claims could be misleading.

GPT-2 was rejected because of frequent hallucinations and poor factual reliability.
