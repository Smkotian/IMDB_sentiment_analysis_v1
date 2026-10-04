"""Stage 4 - small classical model comparison. 5-fold CV on TRAIN only; small fixed grids.
Features = the representation selected in stage 3."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *

def main():
    cfg = load_config()
    f = cfg.get("features", {"ngram": [1, 3], "preproc": "basic", "rep": "binary"})
    ng = tuple(f["ngram"])
    X, y = get_hashed("train", f["preproc"], ng)
    print(f"  features: ngram={ng} rep={f['rep']} preprocessing={f['preproc']}")

    def fs(mf):
        return feat_spec(ng, f["preproc"], f["rep"], mf)

    tree_p = {"n_estimators": 100, "max_features": "sqrt"}
    gb_p = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "max_features": "sqrt"}
    grid = [("logreg", [{"C": c} for c in (0.1, 1, 10)], MAX_FEATURES, N_JOBS),
            ("linear_svm", [{"C": c} for c in (0.01, 0.1, 1)], MAX_FEATURES, N_JOBS),
            ("naive_bayes", [{"alpha": a} for a in (0.1, 0.5, 1.0)], MAX_FEATURES, N_JOBS),
            ("knn", [{"n_neighbors": k} for k in (15, 31)], MAX_FEATURES, 1),
            ("random_forest", [tree_p], MAX_FEATURES_TREES, 1),
            ("gradient_boosting", [gb_p], MAX_FEATURES_TREES, 1)]
    rows, best_by_model = [], {}
    for name, plist, mf, nj in grid:
        for p in plist:
            spec = {"model": name, "params": p, "features": fs(mf)}
            try:
                t0 = time.time()
                s = cv_scores(spec, X, y, n_jobs=nj)
                row = dict(stage="classical", **describe(spec), cv_accuracy=s.mean(), cv_std=s.std(),
                           training_time=(time.time() - t0) / 5)
                rows.append(row)
                print(f"  {name:18s} {p}  cv={s.mean():.4f} +/- {s.std():.4f}")
                if name not in best_by_model or s.mean() > best_by_model[name][0]:
                    best_by_model[name] = (s.mean(), s.std(), spec)
            except Exception as e:  # keep going if one model fails
                print(f"  {name} {p} FAILED: {e}")
                rows.append(dict(stage="classical", **describe(spec), notes=f"failed: {e}"))

    # simple soft-voting ensemble of the best LR and best NB
    if "logreg" in best_by_model and "naive_bayes" in best_by_model:
        vp = {"C": best_by_model["logreg"][2]["params"]["C"], "alpha": best_by_model["naive_bayes"][2]["params"]["alpha"]}
        spec = {"model": "voting_lr_nb", "params": vp, "features": fs(MAX_FEATURES)}
        s = cv_scores(spec, X, y, n_jobs=N_JOBS)
        rows.append(dict(stage="classical", **describe(spec), cv_accuracy=s.mean(), cv_std=s.std(),
                         notes="soft voting: LogisticRegression + MultinomialNB"))
        print(f"  voting_lr_nb       {vp}  cv={s.mean():.4f} +/- {s.std():.4f}")
        best_by_model["voting_lr_nb"] = (s.mean(), s.std(), spec)

    df = pd.DataFrame(rows)
    df["best_for_model"] = False
    for name, (_, _, spec) in best_by_model.items():
        m = (df.model == name) & (df.parameters == describe(spec)["parameters"])
        df.loc[m, "best_for_model"] = True
    save_results("classical_models.csv", df.to_dict("records"))

    top = max(best_by_model.items(), key=lambda kv: kv[1][0])
    update_config("classical", {"name": top[0], "spec": top[1][2], "cv_accuracy": float(top[1][0])})
    print(f"  best classical model (CV): {top[0]}  cv={top[1][0]:.4f}")

    plt = plot_style()
    items = sorted(best_by_model.items(), key=lambda kv: kv[1][0])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh([k for k, _ in items], [v[0] for _, v in items], xerr=[v[1] for _, v in items], color="#4C78A8")
    ax.set_xlim(max(0, min(v[0] for _, v in items) - 0.05), 1.0)
    ax.set_xlabel("5-fold CV accuracy (train only)")
    ax.set_title("Classical models (best setting each)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "classical_model_comparison.png")

if __name__ == "__main__":
    main()
