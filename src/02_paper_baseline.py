"""Stage 2 - PAPER REPRODUCTION: binary 1-3 gram features -> ANN Input-30-30-20-10-10-output.
Uses TRAIN data only (80/20 hold-out + 5-fold CV). The test set is evaluated once, in stage 07."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *

def main():
    spec = PAPER_SPEC
    X, y = get_features_for(spec)
    print("  hold-out validation ...")
    h = holdout_eval(spec, X, y)
    print(f"    val_accuracy={h['val_accuracy']:.4f}  epochs={h['n_epochs']}  time={h['training_time']:.1f}s")
    print("  5-fold CV (train only) ...")
    cv = cv_scores(spec, X, y)
    print(f"    cv_accuracy={cv.mean():.4f} +/- {cv.std():.4f}")
    row = dict(stage="PAPER REPRODUCTION", **describe(spec), cv_accuracy=cv.mean(), cv_std=cv.std(),
               val_accuracy=h["val_accuracy"], training_time=h["training_time"],
               notes="Architecture/features from the brief; activation/optimizer/lr/batch are ASSUMPTIONS")
    save_results("paper_baseline.csv", [row])
    update_config("paper", {"spec": spec, "cv_accuracy": float(cv.mean())})

if __name__ == "__main__":
    main()
