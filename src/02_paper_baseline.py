"""Stage 2 - PAPER-INSPIRED BASELINE: binary 1-3 gram features
-> ANN with hidden layers 30-30-20-10-10.

This is an inspired-by baseline, not an exact reproduction of Wu & Shin.
Uses the supplied aclImdb train split with an 80/20 validation split
and 5-fold CV on training data. The test set is evaluated in stage 7.
"""

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
    row = dict(stage="PAPER INSPIRED BASELINE", **describe(spec), cv_accuracy=cv.mean(), cv_std=cv.std(),
               val_accuracy=h["val_accuracy"], training_time=h["training_time"],
               notes="Architecture/features from the brief; activation/optimizer/lr/batch are ASSUMPTIONS")
    save_results("paper_baseline.csv", [row])
    update_config("paper", {"spec": spec, "cv_accuracy": float(cv.mean())})

if __name__ == "__main__":
    main()
