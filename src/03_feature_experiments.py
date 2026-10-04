"""Stage 3 - controlled feature experiments (fast proxy model: Logistic Regression, 5-fold CV on TRAIN).
A-D n-gram ranges x {binary, count, tf-idf}; then ONE preprocessing change: negation marking."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *

LR = ("logreg", {"C": 1.0})

def run(preproc, ngram, rep):
    spec = {"model": LR[0], "params": LR[1], "features": feat_spec(ngram, preproc, rep)}
    X, y = get_hashed("train", preproc, ngram)
    t0 = time.time()
    s = cv_scores(spec, X, y, n_jobs=N_JOBS)
    row = dict(stage="feature_experiment", **describe(spec), cv_accuracy=s.mean(), cv_std=s.std(),
               training_time=(time.time() - t0) / 5, notes="proxy model = LogisticRegression C=1")
    print(f"  {preproc:8s} ngram={ngram} {rep:6s} cv={s.mean():.4f}")
    return row

def main():
    rows = [run("basic", ng, rep) for ng in NGRAM_RANGES for rep in ("binary", "count", "tfidf")]
    df = pd.DataFrame(rows)
    best_rep = df.loc[df.cv_accuracy.idxmax(), "feature_type"]
    print(f"  best representation (basic preprocessing): {best_rep}; testing negation-aware preprocessing")
    rows += [run("negation", ng, best_rep) for ng in NGRAM_RANGES]
    df = save_results("feature_experiments.csv", rows)
    best = df.loc[df.cv_accuracy.astype(float).idxmax()]
    ng = [int(c) for c in best.ngram_range.split("-")]
    update_config("features", {"ngram": ng, "preproc": best.preprocessing, "rep": best.feature_type,
                               "cv_accuracy": float(best.cv_accuracy)})
    print(f"  selected features (by CV): {best.ngram_range} {best.feature_type} {best.preprocessing}")

    # free disk: keep only hashed matrices that later stages need
    keep = {hashed_path("train", best.preprocessing, tuple(ng)), hashed_path("train", "basic", (1, 3))}
    for p in CACHE_DIR.glob("hash_train_*.npz"):
        if p not in keep:
            p.unlink()

    plt = plot_style()
    df["cv_accuracy"] = df.cv_accuracy.astype(float)
    df["label"] = df.apply(lambda r: r.feature_type + ("+negation" if r.preprocessing == "negation" else ""), axis=1)
    piv = df.pivot(index="ngram_range", columns="label", values="cv_accuracy")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    piv.plot.bar(ax=ax, rot=0)
    lo = float(piv.min().min())
    ax.set_ylim(max(0, lo - 0.02), min(1.0, float(piv.max().max()) + 0.01))
    ax.set_ylabel("5-fold CV accuracy (train only)")
    ax.set_xlabel("n-gram range")
    ax.set_title("Feature experiments (Logistic Regression proxy)")
    ax.legend(title="representation", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_comparison.png")

if __name__ == "__main__":
    main()
