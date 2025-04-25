# 🇺🇦 LLM Alignment for the Ukrainian Language

This project both creates Ukrainian alignment datasets and evaluates how well various open-source language models align with Ukrainian ethical and social norms. It translates established benchmarks to Ukrainian through a multistage adaptation pipeline, then uses these datasets to measure the models' ability to reason ethically and understand social conventions in Ukrainian language contexts.

> ⚠️ **Disclaimer**: The datasets used in this project contain ethical scenarios and examples of social norms, including potentially offensive, harmful, or illegal behavior. These materials are used solely for research and evaluation purposes.

## 📋 Project Overview

The project implements:

- Ukrainian dataset adaptation through multi-stage translation pipeline
- Evaluation frameworks for:
  - Ethical reasoning (adapted from [ETHICS dataset](https://huggingface.co/datasets/hendrycks/ethics))
  - Social norms understanding (adapted from [Social-Chem-101 dataset](https://github.com/mbforbes/social-chemistry-101))
  - Ukrainian-specific evaluations ([Aya-evaluation-suite](https://huggingface.co/datasets/CohereLabs/aya_evaluation_suite/viewer/dolly_machine_translated?views%5B%5D=dolly_machine_translated))
  - Mixed passive value alignment (PVA) evaluations

## 🤖 Models Evaluated

- [Aya models](https://cohere.com/research/aya) (101, Expanse)
- [Llama 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [Gemma](https://ai.google.dev/gemma/docs/core) (2 and 3)
- [Qwen](https://github.com/QwenLM/Qwen)
- [GPT-4o](https://openai.com/index/hello-gpt-4o/) (for comparison)

## ✨ Features

- Comprehensive evaluation metrics: hard accuracy, soft accuracy, F1 score
- Multiple translation approaches: [DeepL](https://www.deepl.com), [Claude](https://www.anthropic.com/claude), [Dragoman](https://huggingface.co/lang-uk/dragoman)
- Translation quality analysis and comparison
- Grammar error correction and refinement tools
- Result visualization via Jupyter notebooks
- Integration with [Langfuse](https://langfuse.com) for experiment tracking

## 🔄 Dataset Adaptation Pipeline

- **Translation Methods**:

  - [DeepL API](https://developers.deepl.com/docs) for batch translation
  - [Claude 3.7](https://www.anthropic.com/claude/sonnet) for higher quality translations
  - Comparative quality analysis showed a preference for Claude in ~35% of cases for the ETHICS subset and ~31% for the Social Chemistry 101 subset

- **Refinement Process**:
  - Grammar error correction via [Spivavtor model](https://huggingface.co/collections/grammarly/spivavtor-660744ab14fdf5e925592dc7)

## 📊 Evaluation Metrics

- **Accuracy Metrics**:

  - Hard accuracy (standard correct predictions)
  - Soft accuracy (allows certain error types)
  - Label-specific accuracy (for targeted evaluation)

- **Classification Metrics**:

  - Macro F1-score across all labels
  - Precision/recall/F1 (specific to morally unacceptable content)
  - Cross-lingual performance comparison

## 🚀 Getting Started

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

## 📁 Project Structure

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

## 📈 Results

Evaluation results are stored in:

- [`src/results/ethics_results.csv`](src/results/ethics_results.csv)
- [`src/results/sc_101_care_harm_results.csv`](src/results/sc_101_care_harm_results.csv)
- [`src/results/ethics_sc101_pva_results.csv`](src/results/ethics_sc101_pva_results.csv)
