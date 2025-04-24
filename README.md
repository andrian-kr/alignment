# LLM Alignment for the Ukrainian Language

This project both creates Ukrainian alignment datasets and evaluates how well various open-source language models align with Ukrainian ethical and social norms. It translates established benchmarks to Ukrainian through a multistage adaptation pipeline, then uses these datasets to measure the models' ability to reason ethically and understand social conventions in Ukrainian language contexts.

## Project Overview

The project implements:

- Ukrainian dataset adaptation through multi-stage translation pipeline
- Evaluation frameworks for:
  - Ethical reasoning (ETHICS dataset)
  - Social norms understanding (Social-Chem-101 dataset)
  - Ukrainian-specific evaluations (AYA-Eval-UKR)
  - Mixed passive value alignment (PVA) evaluations

## Models Evaluated

- Aya models (101, Expanse)
- Llama 3.2
- Gemma (2 and 3)
- Qwen
- GPT-4o (for comparison)

## Features

- Comprehensive evaluation metrics: hard accuracy, soft accuracy, F1 score
- Multiple translation approaches: DeepL, Claude, Dragoman
- Translation quality analysis and comparison
- Grammar error correction and refinement tools
- Result visualization via Jupyter notebooks
- Integration with Langfuse for experiment tracking

## Dataset Adaptation Pipeline

- **Translation Methods**:

  - DeepL API for batch translation
  - Claude 3.7 for higher quality translations
  - Comparative quality analysis showed a preference for Claude in ~35% of cases for the ETHICS subset and ~31% for the Social Chemistry 101 subset

- **Refinement Process**:
  - Grammar error correction via Spivavtor model

## Evaluation Metrics

- **Accuracy Metrics**:

  - Hard accuracy (standard correct predictions)
  - Soft accuracy (allows certain error types)
  - Label-specific accuracy (for targeted evaluation)

- **Classification Metrics**:
  - Macro F1-score across all labels
  - Bad precision/recall/F1 (specific to morally unacceptable content)
  - Cross-lingual performance comparison

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file based on `.env.example` with necessary API keys:

```
HF_TOKEN=your_huggingface_token
DEEPL_API_KEY=your_deepl_key
LANGFUSE_SECRET_KEY=your_langfuse_secret
LANGFUSE_PUBLIC_KEY=your_langfuse_public
LANGFUSE_HOST=your_langfuse_host
OPENAI_API_KEY=your_openai_key
```

### Running Evaluations

```bash
python -m src.evaluate
```

## Project Structure

- `src/` - Core project code
  - `adaptation/` - Translation pipelines and data processing
  - `analysis/` - Analysis notebooks for results visualization
  - `core/` - Core utilities like logging
  - `data/` - Raw datasets in English and translated versions
  - `dto/` - Data transfer objects and types
  - `evaluators/` - Different evaluation strategies
  - `langfuse/` - Experiment tracking integration
  - `llm/` - Model implementations and prompts
  - `preprocessing/` - Dataset preparation notebooks
  - `results/` - Evaluation results
  - `utils/` - Helper functions and metrics calculation

## Results

Evaluation results are stored in:

- `src/results/ethics_results.csv`
- `src/results/sc_101_care_harm_results.csv`
- `src/results/ethics_sc101_pva_results.csv`
