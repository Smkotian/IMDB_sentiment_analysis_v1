"""Stage 7 - final model selection (TRAIN CV only) + the ONE evaluation on the untouched 25K test set.

Selection rule: among the candidates {paper ANN, paper ANN + best features, best ANN, best classical}
pick the highest 5-fold CV accuracy measured on TRAIN.  Each distinct model is fitted on the full train set and
evaluated on test exactly once; test numbers are only REPORTED, never used to choose anything."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *

def main():
    cfg = load_config()
    paper = cfg.get("paper", {"spec": PAPER_SPEC, "cv_accuracy": float("nan")})
    cands = [("paper ANN", paper["spec"], paper["cv_accuracy"])]
    for key, label in (("feature_improved_ann", "paper ANN + best features"), ("ann", "best ANN"),
                       ("classical", "best classical")):
        if key in cfg:
            cands.append((label if key != "classical" else f"best classical ({cfg[key]['name']})",
                          cfg[key]["spec"], cfg[key]["cv_accuracy"]))
    valid = [c for c in cands if np.isfinite(c[2])]
    final = max(valid, key=lambda c: c[2]) if valid else cands[0]
    print("  candidates (CV accuracy on TRAIN):")
    for c in cands:
        print(f"    {c[0]:40s} {c[2]:.4f}")
    print(f"  -> FINAL MODEL selected by CV: {final[0]}")

    fi = next((c for c in cands if c[0] == "paper ANN + best features"), None)
    roles = [("PAPER REPRODUCTION", cands[0])]
    if fi is not None and json.dumps(fi[1], sort_keys=True) != json.dumps(cands[0][1], sort_keys=True):
        roles.append(("OUR MINIMAL IMPROVEMENT - step 1: features only (paper ANN + best features)", fi))
    roles.append(("OUR MINIMAL IMPROVEMENT - FINAL MODEL (selected by CV)", final))

    evaluated, rows, roc_data, final_pipe, final_cm = {}, [], {}, None, None
    for role, (label, spec, cv_acc) in roles:
        key = json.dumps(spec, sort_keys=True)
        if key not in evaluated:
            f = spec["features"]
            Xtr, ytr = get_hashed("train", f["preproc"], tuple(f["ngram"]))
            Xte, yte = get_hashed("test", f["preproc"], tuple(f["ngram"]), allow_test=True)   # test used HERE ONLY
            pipe = build_pipeline(spec)
            t0 = time.time(); pipe.fit(Xtr, ytr); tt = time.time() - t0
            t1 = time.time(); pred = pipe.predict(Xte); it = time.time() - t1
            sc = get_scores(pipe, Xte)
            m = compute_metrics(yte, pred, sc)
            evaluated[key] = (m, tt, it, pipe, sc, yte)
            print(f"  TEST [{label}]  acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  rec={m['recall']:.4f}  "
                  f"spec={m['specificity']:.4f}  auc={m['roc_auc'] if isinstance(m['roc_auc'], str) else round(m['roc_auc'], 4)}")
        m, tt, it, pipe, sc, yte = evaluated[key]
        rows.append(dict(stage=role, **describe(spec), cv_accuracy=cv_acc, **m, training_time=tt,
                         inference_time=it, notes=f"candidate label: {label}"))
        if sc is not None:
            roc_data[role] = (yte, sc)
        if role.endswith("FINAL MODEL (selected by CV)"):
            final_pipe, final_spec, final_m = pipe, spec, m
    save_results("final_results.csv", rows)

    # save final model (raw text in -> prediction out)
    f = final_spec["features"]
    joblib.dump(SentimentModel(final_pipe, f["preproc"], f["ngram"], description=final[0]),
                MODELS_DIR / "final_model.joblib")
    (MODELS_DIR / "final_model_info.json").write_text(json.dumps({"label": final[0], "spec": final_spec}, indent=2))

    # ---- plots
    plt = plot_style()
    cm = np.array([[final_m["tn"], final_m["fp"]], [final_m["fn"], final_m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["neg", "pos"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["neg", "pos"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix - test set\nacc={final_m['accuracy']:.4f}")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "confusion_matrix.png")

    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for role, (yt, sc) in roc_data.items():
        fpr, tpr, _ = roc_curve(yt, sc)
        ax.plot(fpr, tpr, label=role.split(" - ")[-1] if "-" in role else role)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_title("ROC - test set")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "roc_curve.png")

    # combined model comparison (CV accuracy on train, like-for-like)
    comp = {}
    cm_path = RESULTS_DIR / "classical_models.csv"
    if cm_path.exists():
        d = pd.read_csv(cm_path)
        d = d[d.best_for_model == True]
        for _, r in d.iterrows():
            if str(r.cv_accuracy) != "not_run":
                comp[r.model] = (float(r.cv_accuracy), float(r.cv_std))
    for label, spec, cv_acc in cands:
        if np.isfinite(cv_acc) and not label.startswith("best classical"):
            comp["ANN: " + label] = (cv_acc, 0.0)
    if comp:
        items = sorted(comp.items(), key=lambda kv: kv[1][0])
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(items) + 1.5))
        ax.barh([k for k, _ in items], [v[0] for _, v in items], xerr=[v[1] for _, v in items], color="#E15759")
        ax.set_xlim(max(0, min(v[0] for _, v in items) - 0.05), 1.0)
        ax.set_xlabel("5-fold CV accuracy (train only)")
        ax.set_title("Model comparison (selection basis; test set not used)")
        fig.tight_layout(); fig.savefig(PLOTS_DIR / "model_comparison.png")

    build_master()
    print("  saved models/final_model.joblib, results/final_results.csv, plots/")

if __name__ == "__main__":
    main()
