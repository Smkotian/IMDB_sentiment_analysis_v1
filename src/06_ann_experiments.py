"""Stage 6 - bounded ANN experiment (own numpy backprop implementation => SGD, Momentum, Adagrad, Adam are all REAL).
Greedy 3-step search, 15 configs max, 80/20 hold-out inside TRAIN:
  A) architecture x activation  (Adam, no L2)            9 runs
  B) optimizer at best (arch, act)  (SGD/Momentum/Adagrad) 3 runs (Adam reused from A)
  C) L2 in {1e-5, 1e-4, 1e-3} at best setting              3 runs
The best config and the 'paper ANN + best features' config then get 5-fold CV for a like-for-like comparison."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *

LRS = {"sgd": 0.5, "momentum": 0.1, "adagrad": 0.05, "adam": 1e-3}
ARCHS = {"paper_30-30-20-10-10": [30, 30, 20, 10, 10], "64-32": [64, 32], "128-64-32": [128, 64, 32]}
ACTS = ["sigmoid", "tanh", "relu"]

def main():
    cfg = load_config()
    f = cfg.get("features", {"ngram": [1, 3], "preproc": "basic", "rep": "binary"})
    ng = tuple(f["ngram"])
    fspec = feat_spec(ng, f["preproc"], f["rep"])
    X, y = get_hashed("train", f["preproc"], ng)
    print(f"  features: ngram={ng} rep={f['rep']} preprocessing={f['preproc']}")
    done, rows = {}, []

    def run(step, arch, act, opt, l2):
        key = (arch, act, opt, l2)
        if key in done:
            return done[key]
        spec = {"model": "ann", "params": {"hidden": ARCHS[arch], "activation": act, "optimizer": opt,
                                           "lr": LRS[opt], "l2": l2, "patience": 5}, "features": fspec}
        try:
            h = holdout_eval(spec, X, y)
            row = dict(stage=f"ann_step_{step}", **describe(spec), val_accuracy=h["val_accuracy"],
                       training_time=h["training_time"], architecture=arch, activation=act,
                       optimizer=opt, l2=l2, n_epochs=h["n_epochs"])
            if h["val_accuracy"] < 0.55:
                print("DID NOT CONVERGE")
                row["notes"] = "did not converge"
            print(f"  [{step}] {arch:22s} {act:8s} {opt:8s} l2={l2:g}  val={h['val_accuracy']:.4f}  ({h['training_time']:.0f}s)")
        except Exception as e:
            row = dict(stage=f"ann_step_{step}", **describe(spec), val_accuracy=float("nan"), notes=f"failed: {e}",
                       architecture=arch, activation=act, optimizer=opt, l2=l2)
            print(f"  [{step}] {key} FAILED: {e}")
        row["_spec"] = spec
        done[key] = row
        rows.append(row)
        save_results("ann_experiments.csv", [{k: v for k, v in r.items() if k != "_spec"} for r in rows])
        return row

    def best_of(keys):
        c = [done[k] for k in keys]
        return max(c, key=lambda r: -1 if not np.isfinite(r["val_accuracy"]) else r["val_accuracy"])

    keysA = [(a, act, "adam", 0.0) for a in ARCHS for act in ACTS]
    for k in keysA:
        run("A", *k)
    keysA_relu = [k for k in keysA if k[1] == "relu"]
    bA = best_of(keysA_relu)
    arch, act = bA["architecture"], bA["activation"]
    keysB = [(arch, act, o, 0.0) for o in ("adam", "sgd", "momentum", "adagrad")]
    for k in keysB:
        run("B", *k)
    bB = best_of(keysB)
    keysC = [(arch, act, bB["optimizer"], l) for l in (0.0, 1e-5, 1e-4, 1e-3)]
    for k in keysC:
        run("C", *k)
    best = best_of(keysC)

    # like-for-like 5-fold CV on TRAIN for (i) paper ANN + selected features, (ii) best ANN
    paper = cfg.get("paper", {"spec": PAPER_SPEC, "cv_accuracy": float("nan")})
    fi_spec = dict(PAPER_SPEC, features=fspec)
    if fspec == PAPER_SPEC["features"]:
        fi_cv = paper["cv_accuracy"]
        print("  selected features == paper features -> feature-only step equals the paper baseline")
    else:
        s = cv_scores(fi_spec, X, y)
        fi_cv = float(s.mean())
        rows.append(dict(stage="ann_cv_feature_improved_paper_ann", **describe(fi_spec), cv_accuracy=s.mean(),
                         cv_std=s.std(), architecture="paper_30-30-20-10-10", activation="relu", optimizer="adam", l2=0.0))
        print(f"  CV paper-ANN + best features: {s.mean():.4f}")
    update_config("feature_improved_ann", {"spec": fi_spec, "cv_accuracy": fi_cv})

    if best["_spec"]["params"] == fi_spec["params"] and fspec == fi_spec["features"]:
        best_cv = fi_cv
    else:
        s = cv_scores(best["_spec"], X, y)
        best_cv = float(s.mean())
        rows.append(dict(stage="ann_cv_best", **describe(best["_spec"]), cv_accuracy=s.mean(), cv_std=s.std(),
                         architecture=best["architecture"], activation=best["activation"],
                         optimizer=best["optimizer"], l2=best["l2"]))
        print(f"  CV best ANN: {s.mean():.4f}")
    update_config("ann", {"spec": best["_spec"], "cv_accuracy": best_cv})
    save_results("ann_experiments.csv", [{k: v for k, v in r.items() if k != "_spec"} for r in rows])

    plt = plot_style()
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "_spec"} for r in rows if r["stage"].startswith("ann_step")])
    d = d.dropna(subset=["val_accuracy"]).drop_duplicates(subset=["architecture", "activation", "optimizer", "l2"])
    d["label"] = d.architecture.str.replace("paper_", "") + " | " + d.activation + " | " + d.optimizer + " | l2=" + d.l2.astype(str)
    d = d.sort_values("val_accuracy")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(d) + 1.5))
    ax.barh(d.label, d.val_accuracy, color="#59A14F")
    ax.set_xlim(max(0, d.val_accuracy.min() - 0.03), min(1, d.val_accuracy.max() + 0.01))
    ax.set_xlabel("hold-out validation accuracy (20% of train)")
    ax.set_title("ANN experiments (bounded greedy search)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ann_experiments.png")

if __name__ == "__main__":
    main()
