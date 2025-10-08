# Expert Identification for Health Claims

This repository contains the implementation of the cross-genre information retrieval system described in the paper "Expert Identification for Health Claims: Cross-Genre Information Retrieval to Combat Medical Misinformation". The system identifies experts who can verify health claims by analyzing the relevance between an expert's published work and the specific health claim.

## Overview

Online platforms have become the main providers of health information, but they can also be sources of health-related rumors and misinformation. This system helps combat medical misinformation by identifying the most appropriate experts to verify health claims. The approach works in two phases:

1. **Initial candidate selection**: Efficiently narrows down the expert pool using various retrieval algorithms
2. **Re-ranking**: Refines the results using transformer-based models to ensure accurate claim-to-expert alignment

## Dataset

Our dataset is publicly available on Dryad:

Dataset Link: https://datadryad.org/stash/dataset/doi:10.5061/dryad.xxxxx

## Repository Structure

The code is organized into two phases:

### Phase 1: Initial Candidate Selection

- `BM25.py`: Implements BM25, BM25+, and BM25L algorithms for initial expert retrieval
- `SBERT.py`: Implements Sentence-BERT models (DistilBERT, MiniLM, PubMedBERT) for semantic retrieval
- `evaluation.py`: Evaluates the retrieval performance using metrics like Precision@k, Recall@k, MRR, and MAP

### Phase 2: Re-ranking

- `train_evaluate.py`: Trains and evaluates cross-encoder models to re-rank the initially retrieved candidates
- `evaluation.py`: Evaluates the final re-ranking performance


## Usage

### Initial Candidate Selection (Phase 1)

#### BM25 Models

To run the BM25-based retrieval:

```bash
python BM25.py --datasets [gold|silver] --model [roberston|bm25+|bm25l]
```

Example:
```bash
python BM25.py --datasets gold --model bm25l
```

#### Sentence-BERT Models

To run the Sentence-BERT-based retrieval:

```bash
python SBERT.py
```

### Re-ranking (Phase 2)

To train and evaluate the cross-encoder models for re-ranking:

```bash
python train_predict.py
```

### Evaluation

To evaluate the results:

```bash
python evaluation.py --datasets [gold|silver] --result_file [result_file_path]
```

Example:
```bash
python evaluation.py --datasets gold --result_file results_step1_gold_bm25l_dict.json
```
