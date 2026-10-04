"""Run the whole pipeline:  python run_all.py   [--force] [--from N]
Stages whose output already exists are skipped (use --force to recompute)."""
import subprocess
import sys
import time
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# (number, title, script, done-marker, critical)
STAGES = [
    (1, "Preparing data", "01_prepare_data.py", "data/cache/test_negation.pkl", True),
    (2, "Running paper baseline (ANN 30-30-20-10-10, binary 1-3 grams)", "02_paper_baseline.py", "results/paper_baseline.csv", False),
    (3, "Running feature experiments", "03_feature_experiments.py", "results/feature_experiments.csv", False),
    (4, "Running classical model comparison", "04_classical_models.py", "results/classical_models.csv", False),
    (5, "Running decision-tree bias/variance", "05_tree_bias_variance.py", "results/tree_bias_variance.csv", False),
    (6, "Running ANN experiments", "06_ann_experiments.py", "results/ann_done.flag", False),
    (7, "Final model selection + single test evaluation", "07_final_evaluation.py", "results/final_results.csv", True),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute every stage")
    ap.add_argument("--from", dest="start", type=int, default=1, help="start at stage N")
    args = ap.parse_args()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    (ROOT / "results").mkdir(exist_ok=True)
    log = open(ROOT / "results" / "run_log.txt", "a", encoding="utf-8")
    t_all, failed = time.time(), []
    for n, title, script, marker, critical in STAGES:
        header = f"[{n}/7] {title}"
        if n < args.start:
            continue
        if not args.force and (ROOT / marker).exists() and n != 7:
            print(f"{header}  -> already done, skipping (use --force to rerun)")
            continue
        print(header, flush=True)
        log.write(header + "\n")
        t0 = time.time()
        proc = subprocess.Popen([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()
        log.flush()
        if proc.returncode != 0:
            failed.append(n)
            msg = f"  !! stage {n} failed (exit code {proc.returncode})"
            print(msg)
            if critical:
                print("  This stage is required - stopping. Fix the error above and rerun `python run_all.py`.")
                sys.exit(1)
            print("  Optional stage - continuing with fallbacks.")
        else:
            if n == 6:
                (ROOT / marker).write_text("done")
            print(f"  done in {time.time() - t0:.0f}s")
    print(f"\nFinished in {(time.time() - t_all) / 60:.1f} min. Failed optional stages: {failed or 'none'}")
    print("Results: results/final_results.csv, results/master_results.csv | Plots: plots/ | Model: models/final_model.joblib")

if __name__ == "__main__":
    main()
