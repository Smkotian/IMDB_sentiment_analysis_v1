"""Stage 5 - Decision Tree depth sweep -> bias/variance (under/over-fitting) plot. TRAIN only, 5-fold CV."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *
from sklearn.model_selection import validation_curve

def main():
    cfg = load_config()
    f = cfg.get("features", {"ngram": [1, 3], "preproc": "basic", "rep": "binary"})
    ng = tuple(f["ngram"])
    X, y = get_hashed("train", f["preproc"], ng)
    spec = {"model": "decision_tree", "params": {}, "features": feat_spec(ng, f["preproc"], f["rep"], MAX_FEATURES_TREES)}
    depths = [3, 5, 10, 15, None]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    tr, va = validation_curve(build_pipeline(spec), X, y, param_name="model__max_depth", param_range=depths,
                              cv=cv, scoring="accuracy", n_jobs=1)
    rows = []
    for d, a, b in zip(depths, tr, va):
        desc = describe(spec)
        desc["parameters"] = f"max_depth={d} max_features={MAX_FEATURES_TREES}"
        rows.append(dict(stage="tree_bias_variance", **desc, cv_accuracy=b.mean(), cv_std=b.std(),
                         train_accuracy=a.mean(), gap_train_minus_val=a.mean() - b.mean()))
        print(f"  max_depth={str(d):5s} train={a.mean():.4f} val={b.mean():.4f}")
    save_results("tree_bias_variance.csv", rows)

    plt = plot_style()
    lab = [str(d) for d in depths]
    x = np.arange(len(depths))
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(x, tr.mean(1), "o-", label="train accuracy")
    ax.plot(x, va.mean(1), "s-", label="validation accuracy (5-fold CV)")
    ax.fill_between(x, va.mean(1) - va.std(1), va.mean(1) + va.std(1), alpha=0.2, color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_xlabel("max_depth (None = unlimited)")
    ax.set_ylabel("accuracy")
    ax.set_title("Decision Tree: bias vs variance")
    ax.annotate("high bias\n(under-fit)", (0, va.mean(1)[0]), textcoords="offset points", xytext=(12, 8), fontsize=8)
    ax.annotate("high variance\n(over-fit gap)", (len(x) - 1, tr.mean(1)[-1]), textcoords="offset points",
                xytext=(-80, -40), fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tree_bias_variance.png")

if __name__ == "__main__":
    main()
