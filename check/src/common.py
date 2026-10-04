"""Shared utilities: paths, preprocessing, feature pipeline, numpy MLP, metrics, result I/O.

Everything here is syllabus-level ML (no pretrained models, no deep NLP).
"""
import os
import re
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------- paths / constants
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "aclImdb" / "aclImdb"
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"
CONFIG_PATH = RESULTS_DIR / "selected_config.json"

SEED = 42
N_HASH = 2 ** 20            # hash space used to count n-grams without building a giant vocabulary dict
MAX_FEATURES = 30000        # top-K most frequent n-grams (by TRAIN document frequency) kept as features
MAX_FEATURES_TREES = 5000   # smaller cap for tree ensembles (speed)
MIN_DF = 2                  # n-gram must appear in >= 2 training documents
N_JOBS = max(1, min(4, (os.cpu_count() or 2) - 1))
NGRAM_RANGES = [(1, 1), (1, 2), (1, 3), (2, 3)]

for _d in (CACHE_DIR, RESULTS_DIR, PLOTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


set_seed()

# ----------------------------------------------------------------------------- preprocessing
_HTML = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:()]")
_PUNCT = set(".,!?;:()")
NEGATORS = {"not", "no", "never", "neither", "nor", "hardly", "barely"}
# words that may sit between a negator and the word it modifies ("not VERY good", "not A good")
FILLERS = {"very", "really", "so", "too", "that", "quite", "extremely", "particularly",
           "especially", "a", "an", "the", "at"}


def _normalise(text):
    t = text.lower().replace("\u2019", "'")
    t = _HTML.sub(" ", t)
    t = t.replace("can't", "can not").replace("cannot", "can not").replace("won't", "will not")
    t = re.sub(r"n't\b", " not", t)
    return t


def basic_preprocess(text):
    """Lower-case, strip HTML (<br />), expand n't -> not, drop punctuation."""
    return re.sub(r"[^a-z0-9' ]+", " ", _normalise(text))


def negation_preprocess(text):
    """Basic cleaning + simple negation marking.
    'not good' -> 'not_good', 'not very good' -> 'not_very_good'.
    A negator is glued to the next word; if that word is a filler (very, a, the ...)
    the following word is glued too (max 4 words in total). Never crosses punctuation.
    """
    toks = _TOKEN.findall(_normalise(text))
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in _PUNCT:
            i += 1
            continue
        if t in NEGATORS and i + 1 < len(toks) and toks[i + 1] not in _PUNCT:
            parts, j = [t], i + 1
            while j < len(toks) and toks[j] not in _PUNCT and len(parts) < 4:
                parts.append(toks[j])
                j += 1
                if parts[-1] not in FILLERS:
                    break
            out.append("_".join(parts))
            i = j
        else:
            out.append(t)
            i += 1
    return " ".join(out)


PREPROCESSORS = {"basic": basic_preprocess, "negation": negation_preprocess}

# ----------------------------------------------------------------------------- data access
def read_raw_split(split):
    """Read data/aclImdb/aclImdb/<split>/{neg,pos}/*.txt -> (list[str], np.array labels)."""
    texts, labels = [], []
    for lab_name, lab in (("neg", 0), ("pos", 1)):
        folder = DATA_DIR / split / lab_name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"IMDb folder not found: {folder}\n"
                f"Place the extracted Stanford 'aclImdb' folder at {DATA_DIR} (see README).")
        for f in sorted(folder.glob("*.txt")):
            texts.append(f.read_text(encoding="utf-8", errors="ignore"))
            labels.append(lab)
    return texts, np.array(labels, dtype=np.int64)


def text_cache_path(split, preproc):
    return CACHE_DIR / f"{split}_{preproc}.pkl"


def load_split(split, preproc="basic", allow_test=False):
    """Load preprocessed texts + labels from the cache built by stage 01.
    The TEST split is blocked unless allow_test=True (only stage 07 does that)."""
    if split == "test" and not allow_test:
        raise PermissionError("Test set is locked: only 07_final_evaluation.py may read it.")
    p = text_cache_path(split, preproc)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run 01_prepare_data.py first.")
    return joblib.load(p)


def hash_matrix(texts, ngram_range):
    """Stateless n-gram counting (no vocabulary is fitted -> cannot leak from test data)."""
    hv = HashingVectorizer(n_features=N_HASH, ngram_range=tuple(ngram_range), alternate_sign=False,
                           norm=None, lowercase=False, dtype=np.float32)
    return hv.transform(texts)


def hashed_path(split, preproc, ngram):
    return CACHE_DIR / f"hash_{split}_{preproc}_{ngram[0]}{ngram[1]}.npz"


def get_hashed(split, preproc, ngram, allow_test=False):
    """Return (X_hashed_counts csr, y). Cached on disk."""
    texts, y = load_split(split, preproc, allow_test=allow_test)
    p = hashed_path(split, preproc, ngram)
    if p.exists():
        return sp.load_npz(p).tocsr(), y
    X = hash_matrix(texts, ngram).tocsr()
    sp.save_npz(p, X, compressed=False)
    return X, y


def get_holdout(y):
    """Fixed stratified 80/20 split of the TRAIN set (used for quick ANN validation)."""
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    return tr, va

# ----------------------------------------------------------------------------- feature transformers
class TopKDocFreq(BaseEstimator, TransformerMixin):
    """Keep the max_features hashed columns with highest document frequency (>= min_df).
    Fitted only on the data it is given (training fold) -> leakage-free inside CV."""

    def __init__(self, max_features=MAX_FEATURES, min_df=MIN_DF):
        self.max_features = max_features
        self.min_df = min_df

    def fit(self, X, y=None):
        X = sp.csr_matrix(X)
        df = np.bincount(X.indices, minlength=X.shape[1])
        cand = np.where(df >= self.min_df)[0]
        if len(cand) == 0:
            cand = np.where(df > 0)[0]
        order = cand[np.argsort(-df[cand], kind="stable")][: self.max_features]
        self.cols_ = np.sort(order)
        self.map_ = -np.ones(X.shape[1], dtype=np.int64)
        self.map_[self.cols_] = np.arange(len(self.cols_))
        return self

    def transform(self, X):
        X = sp.csr_matrix(X)
        new = self.map_[X.indices]
        keep = new >= 0
        rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))[keep]
        return sp.csr_matrix((X.data[keep], (rows, new[keep])),
                             shape=(X.shape[0], len(self.cols_)), dtype=np.float32)


def feature_steps(rep, max_features):
    steps = [("select", TopKDocFreq(max_features=max_features))]
    if rep == "binary":
        steps.append(("binarize", Binarizer()))
    elif rep == "tfidf":
        steps.append(("tfidf", TfidfTransformer(sublinear_tf=True)))
    elif rep != "count":
        raise ValueError(rep)
    return steps

# ----------------------------------------------------------------------------- numpy MLP (backprop)
def _act(name, z):
    if name == "relu":
        return np.maximum(z, 0)
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    raise ValueError(name)


def _dact(name, a):
    """Derivative expressed through the activation OUTPUT a."""
    if name == "relu":
        return (a > 0).astype(a.dtype)
    if name == "tanh":
        return 1.0 - a * a
    return a * (1.0 - a)


class NumpyMLP(BaseEstimator, ClassifierMixin):
    """Small fully-connected network trained with mini-batch backprop (binary cross-entropy,
    sigmoid output, optional L2). Accepts scipy sparse input.
    optimizer: 'sgd' | 'momentum' | 'adagrad' | 'adam'  (all implemented here by hand).
    Early stopping on an internal validation slice of the data it is given."""

    def __init__(self, hidden=(30, 30, 20, 10, 10), activation="relu", optimizer="adam", lr=1e-3,
                 l2=0.0, momentum=0.9, batch_size=256, max_epochs=20, patience=3,
                 val_fraction=0.1, random_state=SEED):
        self.hidden = hidden
        self.activation = activation
        self.optimizer = optimizer
        self.lr = lr
        self.l2 = l2
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    # --- init / forward / backward
    def _init_params(self, n_in, rng):
        sizes = [n_in, *self.hidden, 1]
        self.W_, self.b_ = [], []
        for i, (fi, fo) in enumerate(zip(sizes[:-1], sizes[1:])):
            if self.activation == "relu" and i < len(sizes) - 2:
                W = rng.randn(fi, fo) * np.sqrt(2.0 / fi)                 # He init
            else:
                lim = np.sqrt(6.0 / (fi + fo))
                W = rng.uniform(-lim, lim, size=(fi, fo))                  # Xavier init
            self.W_.append(W.astype(np.float32))
            self.b_.append(np.zeros(fo, dtype=np.float32))

    def _forward(self, X):
        acts = [X]
        a = X
        L = len(self.W_)
        for i in range(L):
            z = np.asarray(a @ self.W_[i]) + self.b_[i]
            a = _act(self.activation if i < L - 1 else "sigmoid", z).astype(np.float32)
            acts.append(a)
        return acts

    def _grads(self, X, y):
        acts = self._forward(X)
        m = X.shape[0]
        delta = (acts[-1] - y.reshape(-1, 1)) / m                          # dL/dz for sigmoid + BCE
        gW, gb = [None] * len(self.W_), [None] * len(self.W_)
        for i in range(len(self.W_) - 1, -1, -1):
            gW[i] = np.asarray(acts[i].T @ delta, dtype=np.float32)
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W_[i].T) * _dact(self.activation, acts[i])
        if self.l2 > 0:
            for i in range(len(gW)):
                gW[i] += self.l2 * self.W_[i]
        return gW, gb

    # --- optimizers
    def _update(self, grads, t):
        params = self.W_ + self.b_
        lr = self.lr
        for k, (p, g) in enumerate(zip(params, grads)):
            if self.optimizer == "sgd":
                p -= lr * g
            elif self.optimizer == "momentum":
                v = self._s1[k]
                v *= self.momentum
                v += g
                p -= lr * v
            elif self.optimizer == "adagrad":
                G = self._s1[k]
                G += g * g
                p -= lr * g / (np.sqrt(G) + 1e-8)
            elif self.optimizer == "adam":
                m, v = self._s1[k], self._s2[k]
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            else:
                raise ValueError(self.optimizer)

    def _proba1(self, X, bs=4096):
        X = sp.csr_matrix(X, dtype=np.float32)
        return np.concatenate([self._forward(X[i:i + bs])[-1].ravel() for i in range(0, X.shape[0], bs)])

    def _loss(self, X, y):
        p = np.clip(self._proba1(X), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # --- sklearn API
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = sp.csr_matrix(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32)
        n = X.shape[0]
        perm = rng.permutation(n)
        nv = int(n * self.val_fraction) if self.val_fraction > 0 else 0
        va, tr = perm[:nv], perm[nv:]
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = (X[va], y[va]) if nv else (None, None)
        self._init_params(X.shape[1], rng)
        params = self.W_ + self.b_
        self._s1 = [np.zeros_like(p) for p in params]
        self._s2 = [np.zeros_like(p) for p in params] if self.optimizer == "adam" else None
        self.classes_ = np.array([0, 1])
        self.history_ = []
        best, best_params, wait, t = np.inf, None, 0, 0
        for ep in range(self.max_epochs):
            order = rng.permutation(Xtr.shape[0])
            for s in range(0, len(order), self.batch_size):
                b = order[s:s + self.batch_size]
                gW, gb = self._grads(Xtr[b], ytr[b])
                t += 1
                self._update(gW + gb, t)
            tr_loss = self._loss(Xtr, ytr)
            va_loss = self._loss(Xva, yva) if nv else tr_loss
            if not np.isfinite(va_loss):
                break
            self.history_.append((ep + 1, tr_loss, va_loss))
            if va_loss < best - 1e-4:
                best, wait = va_loss, 0
                best_params = [p.copy() for p in self.W_ + self.b_]
            else:
                wait += 1
                if nv and wait >= self.patience:
                    break
        if best_params is not None:
            L = len(self.W_)
            self.W_, self.b_ = best_params[:L], best_params[L:]
        self.n_epochs_ = len(self.history_)
        return self

    def predict_proba(self, X):
        p = self._proba1(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._proba1(X) >= 0.5).astype(int)

# ----------------------------------------------------------------------------- model specs
def make_model(name, params):
    p = dict(params)
    if name == "ann":
        p["hidden"] = tuple(p.get("hidden", (30, 30, 20, 10, 10)))
        return NumpyMLP(**p)
    if name == "logreg":
        return LogisticRegression(solver="liblinear", max_iter=1000, random_state=SEED, **p)
    if name == "linear_svm":
        return LinearSVC(random_state=SEED, max_iter=5000, **p)
    if name == "naive_bayes":
        return MultinomialNB(**p)
    if name == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **p)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=SEED, **p)
    if name == "knn":
        return KNeighborsClassifier(metric="cosine", algorithm="brute", n_jobs=N_JOBS, **p)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=SEED, **p)
    if name == "voting_lr_nb":
        return VotingClassifier([("lr", LogisticRegression(solver="liblinear", C=p.get("C", 1.0),
                                                           random_state=SEED)),
                                 ("nb", MultinomialNB(alpha=p.get("alpha", 1.0)))], voting="soft")
    raise ValueError(name)


def feat_spec(ngram, preproc, rep, max_features=MAX_FEATURES):
    return {"ngram": list(ngram), "preproc": preproc, "rep": rep, "max_features": max_features}


def build_pipeline(spec):
    """spec = {'model': name, 'params': {...}, 'features': feat_spec(...)} -> sklearn Pipeline
    that starts from HASHED COUNTS (see get_hashed)."""
    f = spec["features"]
    steps = feature_steps(f["rep"], f["max_features"])
    steps.append(("model", make_model(spec["model"], spec.get("params", {}))))
    return Pipeline(steps)


PAPER_SPEC = {
    "model": "ann",
    # ASSUMPTIONS (paper details not available): ReLU hidden, sigmoid output, Adam lr=1e-3,
    # BCE loss, no L2, mini-batch 256, early stopping.  Architecture + features are from the brief.
    "params": {"hidden": [30, 30, 20, 10, 10], "activation": "relu", "optimizer": "adam",
               "lr": 1e-3, "l2": 0.0},
    "features": feat_spec((1, 3), "basic", "binary"),
}

# ----------------------------------------------------------------------------- metrics / timing
def get_scores(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        pass
    try:
        return model.decision_function(X)
    except Exception:
        return None


def compute_metrics(y, pred, score=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "roc_auc": roc_auc_score(y, score) if score is not None else "not_run",
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def cv_scores(spec, X, y, n_jobs=1):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_score(build_pipeline(spec), X, y, cv=cv, scoring="accuracy", n_jobs=n_jobs)


def holdout_eval(spec, X, y):
    """Fit on 80% of TRAIN, report accuracy on the other 20% of TRAIN."""
    tr, va = get_holdout(y)
    pipe = build_pipeline(spec)
    t0 = time.time()
    pipe.fit(X[tr], y[tr])
    train_time = time.time() - t0
    pred = pipe.predict(X[va])
    return {"val_accuracy": accuracy_score(y[va], pred), "training_time": train_time,
            "n_epochs": getattr(pipe.named_steps["model"], "n_epochs_", "not_run")}

# ----------------------------------------------------------------------------- results I/O
RESULT_COLS = ["stage", "model", "feature_type", "ngram_range", "preprocessing", "parameters",
               "cv_accuracy", "cv_std", "val_accuracy", "accuracy", "precision", "recall",
               "specificity", "roc_auc", "training_time", "inference_time", "notes"]


def save_results(filename, rows):
    df = pd.DataFrame(rows)
    for c in RESULT_COLS:
        if c not in df.columns:
            df[c] = "not_run"
    extra = [c for c in df.columns if c not in RESULT_COLS]
    df = df[RESULT_COLS + extra].fillna("not_run")
    df.to_csv(RESULTS_DIR / filename, index=False)
    return df


def load_config():
    return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}


def update_config(key, value):
    cfg = load_config()
    cfg[key] = value
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def describe(spec):
    f = spec["features"]
    return dict(model=spec["model"], feature_type=f["rep"],
                ngram_range=f"{f['ngram'][0]}-{f['ngram'][1]}", preprocessing=f["preproc"],
                parameters=json.dumps(spec.get("params", {}), sort_keys=True) +
                           f" max_features={f['max_features']}")


def get_features_for(spec):
    f = spec["features"]
    return get_hashed("train", f["preproc"], tuple(f["ngram"]))


class SentimentModel:
    """Raw text -> prediction. Bundles preprocessing + hashing + fitted pipeline for saving."""

    def __init__(self, pipeline, preproc, ngram, description=""):
        self.pipeline, self.preproc, self.ngram, self.description = pipeline, preproc, tuple(ngram), description

    def _x(self, texts):
        pre = PREPROCESSORS[self.preproc]
        return hash_matrix([pre(t) for t in texts], self.ngram)

    def predict(self, texts):
        return self.pipeline.predict(self._x(texts))

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(self._x(texts))


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    return plt


def build_master():
    """Concatenate every stage CSV into results/master_results.csv."""
    parts = []
    for name in ("paper_baseline", "feature_experiments", "classical_models", "tree_bias_variance",
                 "ann_experiments", "final_results"):
        p = RESULTS_DIR / f"{name}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d.insert(0, "source_file", p.name)
            parts.append(d)
    if parts:
        pd.concat(parts, ignore_index=True).fillna("not_run").to_csv(RESULTS_DIR / "master_results.csv", index=False)
