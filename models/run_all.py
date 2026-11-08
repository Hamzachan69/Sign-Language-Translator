"""
Run the landmark classifiers.

  python models/run_all.py              # all five
  python models/run_all.py --model mlp
  python models/run_all.py --model svm   (also: knn, kmeans, mlp, cnn)
"""
import argparse, sys, importlib, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELS = {"knn": "knn_classifier", "kmeans": "kmeans_classifier", "svm": "svm_classifier",
          "mlp": "mlp_classifier", "cnn": "cnn_classifier"}


def main():
    ap = argparse.ArgumentParser(description="Train ASL landmark classifiers (KNN, K-Means, SVM, MLP, 1D-CNN)")
    ap.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    args, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest          # forward --limit / --rebuild to the trainers

    t_all = time.time()
    for name in (MODELS if args.model == "all" else [args.model]):
        print(f"\n>>> Starting {name.upper()} ...\n")
        t0 = time.time()
        importlib.import_module(MODELS[name]).main()
        print(f"\n<<< {name.upper()} finished in {(time.time()-t0)/60:.1f} min\n")
    print(f"\n{'='*60}\n  All done in {(time.time()-t_all)/60:.1f} min. Results -> models/results/\n{'='*60}\n")


if __name__ == "__main__":
    main()
