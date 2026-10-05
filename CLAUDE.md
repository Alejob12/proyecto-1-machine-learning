# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Course project for *Aprendizaje de Máquina 2026-20* (Universidad de los Andes): a Kaggle competition classifying the **overall sentiment of Spanish product reviews** into `negativo`, `neutral`, or `positivo`. The full assignment spec is [proyecto202620.pdf](proyecto202620.pdf) (in Spanish) — consult it for rubric and deadline details.

Notebooks live in the **project root** (e.g. `parte1.ipynb`, later `parte2.ipynb`), not in a subfolder, and use paths relative to the root (`data/`, `submissions/`, `models/`). Plans: [PLAN_PARTE1.md](PLAN_PARTE1.md) (current focus) and [PLAN.md](PLAN.md) (both parts).

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pins scikit-learn==1.9.1 so the saved .joblib reloads identically
jupyter notebook parte1.ipynb
# Run the whole notebook headless to verify it end to end:
jupyter nbconvert --to notebook --execute parte1.ipynb --output /tmp/parte1_run.ipynb
```

The system `python3` (Homebrew 3.14) has no scikit-learn; always use the venv.

## Data (`data/`)

- `train.csv` — 12,000 rows, columns `id,text,label`. Classes: positivo 4212, negativo 4212, neutral 3576.
- `eval.csv` — 3,000 rows, columns `id,text` (no labels). This is the Kaggle test set.
- `sample_submission.csv` — submission format: `id,answer` with one label per eval `id`. Kaggle metric is **accuracy**.

Text characteristics that matter for modeling:
- Reviews are ~54 words on average (3–110). About two-thirds use Spanish accents/ñ and the rest are unaccented, so accent normalization is a relevant preprocessing decision.
- Many reviews contain filler/logistics sentences (shipping, where it's kept) plus both praise and complaints. The label is the **global** sentiment, which depends on clause order (recency), negation, and contrastive connectors ("aun así", "pero", "al final de cuentas") — not on counting polar words. E.g. a review with a negative early clause and a positive final clause is labeled `positivo`.

## Hard constraints by stage

Violating these yields a grade of 0 for that stage, so check them before proposing any approach.

**Part 1 — classical ML (closes week 9, deliverables week 11)**
- Only scikit-learn classifiers. No deep learning, no transformers, no pretrained neural embeddings (no word2vec/fastText/BERT vectors).
- Text representation limited to course techniques: bag-of-words, TF-IDF, n-grams (hand-crafted features from these are fine).
- Deliverables: one Jupyter notebook implementing and training the final model + the trained model saved with `pickle` or `joblib`. It must be the group's best public-leaderboard submission and fully reproducible (fix random seeds).
- At least 5 *different* Kaggle submissions are required for participation credit.

**Part 2 — deep learning (closes week 16, deliverables week 14)**
- Deep architectures are mandatory (MLP, CNN, RNN/LSTM/GRU, transfer learning and pretrained models/embeddings allowed and encouraged).
- At least one data augmentation strategy is mandatory, and the notebook must compare performance **with vs. without** augmentation.
- External data only if licensed for redistribution and cited (source, link, license). Synthetic data allowed if its construction is described in detail.
- **Never train on `eval.csv`** (strictly prohibited — this includes pseudo-labeling or fitting vectorizers/tokenizers on it in a way that leaks into training).
- Deliverables: notebook (with evidence of iteration and justification of decisions), model saved via tensorflow/keras or similar, and a document listing external data references and how generative AI was used in the project.

## Working conventions

- `parte1.ipynb` is a skeleton (section titles and text, empty code cells) that the user fills in themselves, step by step. Do not write solution code into the notebook unless the user explicitly asks for a specific piece; explain or review instead.
- Notebooks are graded on documentation (15%) and process quality (15%), so notebooks should explain decisions in markdown cells, not just contain code.
- Custom transformers/functions used inside a pipeline must be defined with `def` inside the notebook itself (no `lambda`, no external `.py`): only the notebook and the `.joblib` are submitted, and the model must reload from them.
- Use a held-out split or cross-validation from `train.csv` for model selection; the public leaderboard is only a partial sample and the private score decides the final ranking.
- Submission files must have exactly the columns `id,answer`, cover all 3,000 eval ids, and use the exact label strings `negativo`/`neutral`/`positivo`.
