# IMDb Movie Review Sentiment Analysis

UE24CS352A Machine Learning, Mini-Project

**Team:** `<Simret M Kotian (PES1UG24CS456)>`, `<Supreeth K (SRN)>`  
**Section:** `H`  
**Problem statement:** `Machine Learning based classification for Sentimental analysis of IMDb reviews`

## Problem statement

Classify IMDb movie reviews as positive or negative. 
The task follows the approach of Wu & Shin, *Machine Learning based
classification for Sentimental analysis of IMDb reviews*: build a paper-style
ANN baseline (binary 1-3 gram features, ANN 30-30-20-10-10, ~90.6% reported),
then try to improve on it with better features, other classical models and a
wider ANN search. 
We follow the paper's approach; we do not claim an exact
replication.

## Dataset

Stanford Large Movie Review Dataset (aclImdb v1, Maas et al., ACL 2011):
25,000 training and 25,000 test reviews, evenly split between positive and
negative. The `unsup/` folder is not used. The dataset is not committed to the
repo.

## Approach

1. **Paper-style baseline.** Binary 1-3 gram features (top 30,000), ANN
   30-30-20-10-10 trained with our own NumPy backprop. Several hyperparameters
   (activation, optimiser, learning rate, batch size) are not given in the
   paper, so we assumed them.
2. **Feature experiments.** N-gram range, binary vs. count vs. TF-IDF, and
   negation handling, compared with 5-fold CV on the training set.
3. **Classical models.** Logistic regression, linear SVM, Naive Bayes, KNN,
   random forest, gradient boosting and a LR+NB voting ensemble.
4. **Decision-tree bias/variance.** Tree depth swept from 3 to unlimited.
5. **ANN experiments.** Architecture and activation (Adam), then optimiser
   (SGD, momentum, Adagrad, Adam), then L2 strength.
6. **Selection and test.** Candidates are ranked by 5-fold CV on the training
   set only. Each is fitted on the full training set and evaluated on the test
   set once, in the last stage. Test numbers are reported, never used to choose
   anything.

## Results

Test accuracy (25,000 reviews) and CV accuracy (training set):

| Model | Features | CV acc. | Test acc. |
|---|---|---|---|
| Paper-style ANN (baseline) | binary 1-3 gram | 88.89% | 89.22% |
| Same ANN + better features | TF-IDF 1-2 gram, negation handling | 90.14% | 90.08% |
| **Final model: logistic regression (C=10)** | TF-IDF 1-2 gram, negation handling | 90.26% | **90.14%** |

Inference:

- Better features gave almost all of the improvement (+0.9 points on test).
  Differences below about 0.2 points are within noise on a 25k test set.
- A linear model on TF-IDF n-grams matched or beat every ANN we tried, so
  logistic regression was selected by CV. The best ANN found by the search
  (64-32, ReLU, Adagrad) scored lower in CV (89.54%).
- Trees did poorly on sparse text. Depth 15 peaked at 72.6% CV, and an
  unlimited tree overfit (100% train, 69.9% CV). Random forest and gradient
  boosting were only run on 5,000 features, so they are handicapped.
- L2 regularisation hurt monotonically (hold-out 89.96% with none, down to
  88.38% at 1e-3), so these networks underfit rather than overfit.
- On a 64-32 ReLU net, hold-out accuracy was Adagrad 89.96%, Adam 89.92%,
  momentum 89.22%, SGD 87.72%. SGD stopped at the 20-epoch cap and is likely
  understated. The hold-out is 5,000 reviews, so Adagrad vs. Adam is a tie.
- The 5-layer sigmoid net stalled at 50% under SGD, momentum and L2 (vanishing
  gradients plus early stopping), so the optimiser and L2 comparisons use ReLU.
  Sigmoid with Adam trained fine.
- Our baseline is 89.22%, about 1.4 points under the paper's ~90.6%. We put
  this down to the hyperparameters the paper does not specify and the 30,000
  feature cap.

Full numbers: `results/final_results.csv` (final) and
`results/master_results.csv` (every experiment). Figures are in `plots/`.

## Setup

Python 3.9 to 3.13.

```bash
git clone <repo-url>
cd IMDB_sentiment_analysis_v1
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux / macOS
pip install -r requirements.txt
```

Download `aclImdb_v1.tar.gz` from https://ai.stanford.edu/~amaas/data/sentiment/
and extract it so the layout is exactly:

```
data/aclImdb/aclImdb/
    train/pos   train/neg
    test/pos    test/neg
```

For example, with the archive in the repo root:

```bash
mkdir -p data/aclImdb
tar -xzf aclImdb_v1.tar.gz -C data/aclImdb
```

## Run

```bash
python run_all.py            # skips stages 2-6 if their outputs already exist
python run_all.py --force    # recompute every stage (full pipeline)
python run_all.py --from 3   # restart from stage 3
```

The repo ships with results from a finished run, so a plain `python run_all.py`
only prepares the data (stage 1) and re-runs the final evaluation (stage 7).
Use `--force` to see the whole pipeline run; the ANN stage is the slowest.
Stages 1 and 7 are required and stop the pipeline if they fail; stages 2-6 are
optional. Terminal logs are appended to `results/run_log.txt`.

```bash
python src/predict.py "A moving, beautifully acted film." "Boring and far too long."
```

## Pipeline

| # | Script (`src/`) | What it does | Output |
|---|---|---|---|
| 1 | `01_prepare_data.py` | Load reviews, clean text, cache splits (including negation-handled text) | `data/cache/` |
| 2 | `02_paper_baseline.py` | Paper-style baseline: binary 1-3 gram, ANN 30-30-20-10-10 | `results/paper_baseline.csv` |
| 3 | `03_feature_experiments.py` | N-gram range, binary/count/TF-IDF, negation handling | `results/feature_experiments.csv` |
| 4 | `04_classical_models.py` | LogReg, SVM, NB, KNN, RF, GB, voting ensemble | `results/classical_models.csv` |
| 5 | `05_tree_bias_variance.py` | Decision-tree depth vs. bias/variance | `results/tree_bias_variance.csv` |
| 6 | `06_ann_experiments.py` | Architecture, activation, optimiser and L2 experiments. Step A runs all activations; Steps B and C continue with ReLU only | `results/ann_experiments.csv` |
| 7 | `07_final_evaluation.py` | Pick the final model by CV, evaluate on test once | `results/final_results.csv`, `models/final_model.joblib` |

`src/common.py` holds the shared code: data loading, hashed n-gram features,
the NumPy MLP (backprop with SGD, momentum, Adagrad, Adam and L2), and the
cross-validation helpers.

## Repository layout

```
run_all.py          single entry point
requirements.txt
src/                stage scripts, common.py, predict.py
results/            CSVs, selected_config.json, run_log.txt
plots/              figures used in the write-up and slides
models/             final_model.joblib, final_model_info.json
data/               aclImdb dataset and caches (not committed)
```

## Reproducibility

- Seed 42 everywhere.
- The test set is never used for selection; it is evaluated in stage 7 only.
- `models/final_model.joblib` was saved with scikit-learn 1.6.1. Keep that
  version (it is pinned in `requirements.txt`), or regenerate the model with
  `python run_all.py --from 7`.

## Reference

Wu & Shin, *Machine Learning based classification for Sentimental analysis of
IMDb reviews*. Dataset: Maas et al., *Learning Word Vectors for Sentiment
Analysis*, ACL 2011.
