"""
==============================================================================
  ASL Classifier - MediaPipe LANDMARKS + K-Nearest Neighbours (KNN)
==============================================================================
HOW TO RUN
----------
  pip install mediapipe scikit-learn opencv-python joblib numpy matplotlib seaborn
  python models/knn_classifier.py [--limit 200] [--rebuild]

PIPELINE
--------
  image -> MediaPipe landmarks -> normalise -> 93 features -> mirror augmentation
        -> StandardScaler -> KNN (k and vote weighting tuned on a validation split)
  No PCA needed: 93 dimensions is already small (the old pixel version used 12 288).

OUTPUT
------
  models/results/knn_model.joblib   {"pipeline", "classes", "name"}
  models/results/knn_confusion_matrix.png

UNINSTALL: delete models/results/knn_*
"""
import sys, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    sys.exit(f"[FAIL] Missing package '{e.name}'.\n"
             f"       Fix: pip install mediapipe scikit-learn opencv-python joblib numpy matplotlib seaborn")

from utils.landmark_data import load_landmark_data, augment, to_features, RESULTS_DIR
from utils.metrics import print_metrics, plot_confusion_matrix

MODEL_PATH = RESULTS_DIR / "knn_model.joblib"
K_VALUES = [3, 5, 7, 9, 15]
WEIGHTS = ["uniform", "distance"]


def make_knn(k, weights):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 60 + "\n  Landmarks + KNN  -  ASL Classifier\n" + "=" * 60)
    X_tr, y_tr, X_te, y_te, classes = load_landmark_data(args.limit, args.rebuild)

    print("\n[2] Tuning k / weights on a 20% validation split ...")
    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
    Xa_f, ya_f = augment(Xa, ya)
    Xa_f = to_features(Xa_f)
    Xv_f = to_features(Xv)
    best, best_acc = None, -1
    for k in K_VALUES:
        for w in WEIGHTS:
            t0 = time.time()
            acc = make_knn(k, w).fit(Xa_f, ya_f).score(Xv_f, yv)
            print(f"       k={k:<2} {w:<9} val accuracy = {acc*100:.2f}%  ({time.time()-t0:.1f}s)")
            if acc > best_acc:
                best, best_acc = (k, w), acc
    print(f"\n  Best: k={best[0]}, weights={best[1]}")

    print("\n[3] Final fit on the full training set ...")
    X_full, y_full = augment(X_tr, y_tr)
    final = make_knn(*best).fit(to_features(X_full), y_full)
    joblib.dump({"pipeline": final, "classes": classes, "name": "KNN"}, MODEL_PATH)
    print(f"       saved {MODEL_PATH}")

    print("\n[4] Test-set evaluation ...\n")
    y_pred = final.predict(to_features(X_te))
    res = print_metrics(y_te, y_pred, classes, f"Landmarks + KNN (k={best[0]})")
    plot_confusion_matrix(res["confusion_matrix"], classes, f"Landmarks + KNN (k={best[0]})",
                          save_path=str(RESULTS_DIR / "knn_confusion_matrix.png"))
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
