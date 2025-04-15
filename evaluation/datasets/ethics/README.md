# ⚠️ Disclaimer

This dataset is provided for **research purposes only**. It may contain ethically sensitive content. Translations were machine-generated and grammar-corrected, and may not fully reflect cultural nuances or ethical standards across regions. **Use with caution.**

---

# ETHICS Commonsense Dataset (Ukrainian Translation)

## Overview

This dataset contains **1700 examples** from the _commonsense_ subset of the [ETHICS dataset](https://huggingface.co/datasets/hendrycks/ethics), translated into Ukrainian. It is intended for evaluating language models on **ethical reasoning tasks** in Ukrainian, supporting research in **cross-lingual moral judgment** and **ethical alignment**.

We selected the **first 1700 examples** from the original dataset due to their shorter text length, making them more suitable for high-quality translation and efficient evaluation.

## Source Dataset

The original ETHICS dataset is available at [hendrycks/ethics](https://huggingface.co/datasets/hendrycks/ethics) on Hugging Face.
It consists of several subsets related to various ethical domains. This translation covers only the **commonsense** subset.

For more details, refer to the paper: [Aligning AI With Shared Human Values](https://openreview.net/forum?id=NkHhVREtA8).

## Dataset Description

The dataset is a single CSV file with the following columns:

- **label**: Binary classification where

  - `0` = ethically acceptable
  - `1` = ethically unacceptable

- **input_en**: Original English ethical scenario

- **input_ukr**: Ukrainian translation of the scenario

### Split Information

This dataset is provided under the `"test"` split to indicate that it is intended for **evaluation only**.

---

## Translation Process

The English scenarios were translated into Ukrainian using the following pipeline:

1. **Initial Translation**: Performed using [**Claude 3.7** (Anthropic)](https://www.anthropic.com/news/claude-3-7-sonnet).
2. **Grammar Correction**: Refined using the [Spivavtor-XXL model](https://huggingface.co/grammarly/spivavtor-xxl), a Ukrainian GEC (Grammatical Error Correction) model.

_Note: No manual or human evaluation was conducted after translation._

---

## Uses

### Direct Use

This dataset can be used to:

- Evaluate **ethical reasoning** in Ukrainian LLMs
- Benchmark **cross-lingual moral judgment**
- Study cultural/linguistic shifts in ethical classification

### Out-of-Scope Use

- Not intended for **training** production systems
- Not intended for **fine-tuning** without appropriate safety reviews
- Not suitable for deployment without acknowledging ethical ambiguity

---

## Dataset Creation

### Curation Rationale

Ukrainian-language resources for ethical AI evaluation are lacking. This translation addresses that gap by making a key English benchmark accessible for Ukrainian-language alignment research.

### Source Data

- The English source data comes from the ETHICS Commonsense subset
- Translations were produced automatically and grammar-corrected using machine learning models

### Annotation

- Original binary labels come from ETHICS
- No new annotations were added during translation

---

## Personal and Sensitive Information

This dataset contains hypothetical ethical scenarios. Some examples may include references to violence, theft, or other morally sensitive behavior. No real personal data is present.

---

## Bias, Risks, and Limitations

- **Cultural Bias**: Ethical norms vary by culture. The original English dataset reflects Western-centric moral reasoning, which may not align with Ukrainian norms even after translation.
- **Translation Bias**: Despite grammar correction, translations may still introduce shifts in meaning or emphasis.
- **No Human Evaluation**: All translations were automated. There was **no human review**, which may result in occasional mistranslations or culturally insensitive wording.
- **Moral Ambiguity**: Ethical judgments are inherently subjective. Binary classification may oversimplify real-world ethical reasoning.

### Recommendations

Use this dataset only for research into cross-lingual ethical reasoning. Avoid deploying models trained or evaluated on this data in sensitive or real-world decision-making contexts without rigorous testing.

---

## Citation

Please cite the original ETHICS dataset:

```bibtex
@inproceedings{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

And the Spivavtor-XXL grammar correction model:

```bibtex
@misc{saini2024spivavtorinstructiontunedukrainian,
      title={Spivavtor: An Instruction Tuned Ukrainian Text Editing Model},
      author={Aman Saini and Artem Chernodub and Vipul Raheja and Vivek Kulkarni},
      year={2024},
      eprint={2404.18880},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.18880},
}
```

## Dataset Card Contact

For any inquiries related to the dataset, please contact:

- **Primary Contact:** Andrian Kravchenko
- **Email:** andriankrav@gmail.com
