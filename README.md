# LLM Alignment for the Ukrainian Language

This project evaluates the alignment of open-source LLMs for the Ukrainian language. It focuses on measuring how well various language models adhere to Ukrainian ethical norms and social conventions.

## Project Overview

The project implements evaluation frameworks for:

- Ethical reasoning (ETHICS dataset)
- Social norms understanding (Social-Chem-101 dataset)
- Ukrainian-specific evaluations (AYA-Eval-UKR)

## Features

- Multiple model support: Aya (101, Expanse), Llama, Gemma, Qwen
- Comprehensive evaluation metrics: hard accuracy, soft accuracy, F1 score
- Translation support using DeepL for cross-lingual evaluation
- Result visualization via Jupyter notebooks

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with the following variables:

```
HF_TOKEN=your_huggingface_token
LANGFUSE_SECRET_KEY=your_langfuse_secret
LANGFUSE_PUBLIC_KEY=your_langfuse_public
LANGFUSE_HOST=your_langfuse_host
```

### Running Evaluations

```bash
python -m evaluation.main
```

## Project Structure

- `evaluation/` - Core evaluation code
  - `core/` - Logger and utilities
  - `datasets/` - Evaluation datasets
  - `evaluators/` - Implementation of different evaluation strategies
  - `llm/` - Model implementations and prompts
  - `utils/` - Metrics and helper functions
- `notebooks/` - Visualization and analysis notebooks
  - `visualization/` - Result visualization

## Results

Evaluation results are stored in:

- `evaluation/results/ethics_results.csv`
- `evaluation/results/sc_101_care_harm_results.csv`
