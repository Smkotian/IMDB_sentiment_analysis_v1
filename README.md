
# IMDb Movie Review Sentiment Analysis

UE24CS352A – Machine Learning, Mini-Project

Binary sentiment classification (positive / negative) of IMDb reviews using only
classical, syllabus-covered ML. The project reproduces the ANN baseline from
Wu & Shin, *"Machine Learning based classification for Sentimental analysis of
IMDb reviews"* (binary 1–3 gram features, ANN 30-30-20-10-10, ~90.6% reported),
then compares feature variants and classical models against it.


## Dataset

Stanford **Large Movie Review Dataset (aclImdb v1)**: 25,000 train / 25,000 test
reviews, balanced. The `unsup/` folder is not used.

Download: https://ai.stanford.edu/~amaas/data/sentiment/ (`aclImdb_v1.tar.gz`)

The dataset is **not** committed. Extract it so the layout is:

```
data/aclImdb/
    train/pos  train/neg
    test/pos   test/neg
```

## Constraints

- Concepts used: Logistic Regression, KNN, Decision Trees, Random Forest,
  Gradient Boosting, Naive Bayes, SVM, MLP with backprop (SGD / Momentum /
  Adagrad / Adam), L2 regularisation, cross-validation.
- No BERT / LSTM / pretrained models / learned embeddings.
- Fixed seed (42). The test set is used once, in the final stage only.

## Setup

Python 3.9+.

```bash
git clone <repo-url>
cd IMDB_sentiment_analysis_v1
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Dependencies: numpy, scipy, pandas, scikit-learn, matplotlib, joblib.

## Run

```bash
python run_all.py            # full pipeline; skips stages whose output already exists
python run_all.py --force    # recompute everything
python run_all.py --from 3   # restart from stage 3
```

Stage 7 always runs. Stages 1 and 7 are critical (pipeline stops on failure);
stages 2–6 are optional and fall back gracefully. All output is also written to
`results/run_log.txt`.

## Pipeline

| # | Script (`src/`) | What it does | Output |
|---|---|---|---|
| 1 | `01_prepare_data.py` | Load reviews, clean text, build cached splits (incl. negation-handled test text) | `data/cache/` |
| 2 | `02_paper_baseline.py` | Paper baseline: binary 1–3 gram features, ANN 30-30-20-10-10 | `results/paper_baseline.csv` |
| 3 | `03_feature_experiments.py` | Feature variants (n-gram range, binary vs. count vs. TF-IDF, negation handling) | `results/feature_experiments.csv` |
| 4 | `04_classical_models.py` | LogReg, KNN, Tree, RF, GB, NB, SVM compared | `results/classical_models.csv` |
| 5 | `05_tree_bias_variance.py` | Decision-tree depth vs. bias/variance | `results/tree_bias_variance.csv` |
| 6 | `06_ann_experiments.py` | MLP architecture / optimiser / L2 experiments | `results/ann_done.flag` |
| 7 | `07_final_evaluation.py` | Select final model on validation/CV, evaluate once on test | `results/final_results.csv`, `models/final_model.joblib` |

## Repository layout

```
run_all.py          # single entry point
requirements.txt
src/                # numbered stage scripts + helpers
results/            # CSVs (paper_baseline, feature_experiments, classical_models,
                    #       tree_bias_variance, master_results, final_results), run_log.txt
plots/              # figures used in the write-up / slides
models/             # final_model.joblib
check/              # sanity-check scripts
```

## Results

Final test metrics: `results/final_results.csv`.
Every experiment in one table: `results/master_results.csv`.
Figures: `plots/`.

## Using the saved model

```python
import joblib
model = joblib.load("models/final_model.joblib")
print(model.predict(["A moving, beautifully acted film."]))
```

> If the saved object is a dict (vectoriser + classifier) rather than a single
> pipeline, load both keys instead — check `07_final_evaluation.py`.

## Reproducibility

Seed 42 everywhere; test set touched only in stage 7; stage outputs are cached,
so reruns are fast.

## Reference

Wu & Shin, *Machine Learning based classification for Sentimental analysis of
IMDb reviews*. Dataset: Maas et al., ACL 2011.
