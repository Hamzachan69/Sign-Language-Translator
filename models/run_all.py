"""
══════════════════════════════════════════════════════════════════════════════
  ASL Hand-Gesture Recognition  –  Run All Three Classifiers
══════════════════════════════════════════════════════════════════════════════

HOW TO RUN
----------
  # run all three algorithms one after another
  python models/run_all.py

  # run a specific algorithm only
  python models/run_all.py --model knn
  python models/run_all.py --model kmeans
  python models/run_all.py --model cnn

WHAT IT DOES
------------
  Calls each classifier in sequence and prints a final comparison table.
"""

import argparse, sys, importlib, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


MODELS = {
    "knn":    "knn_classifier",
    "kmeans": "kmeans_classifier",
    "cnn":    "cnn_classifier",
}


def run_model(name: str):
    print(f"\n>>> Starting {name.upper()} ...\n")
    t0  = time.time()
    mod = importlib.import_module(MODELS[name])
    mod.main()
    elapsed = time.time() - t0
    print(f"\n<<< {name.upper()} finished in {elapsed/60:.1f} min\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run ASL classifiers (KNN, K-Means, CNN)"
    )
    parser.add_argument(
        "--model", choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model to run (default: all)",
    )
    args = parser.parse_args()

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]

    total_start = time.time()
    for name in targets:
        run_model(name)

    total = time.time() - total_start
    print(f"\n{'═'*60}")
    print(f"  All done in {total/60:.1f} min.")
    print(f"  Results → models/results/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
