"""
==============================================================================
  ASL Classifier - MediaPipe LANDMARKS + K-Means (unsupervised + majority vote)
==============================================================================
K-Means never sees labels while clustering. Afterwards each cluster is named by
the most common true letter inside it (majority vote), and predictions are the
label of the nearest cluster.

HOW TO RUN
----------
  pip install mediapipe scikit-learn opencv-python joblib numpy matplotlib seaborn
  python models/kmeans_classifier.py [--limit 200] [--rebuild]

PIPELINE
--------
  image -> MediaPipe landmarks -> normalise -> 93 features -> mirror augmentation
        -> StandardScaler -> MiniBatchKMeans(K clusters) -> majority-vote label map
  K is tried at 36 / 108 / 216 clusters (36 = one per class; more = several per class).

OUTPUT
------
  models/results/kmeans_model.joblib   {"pipeline", "classes", "name"}
  models/results/kmeans_confusion_matrix.png

UNINSTALL: delete models/results/kmeans_*
"""
import sys, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
except ImportError as e:
    sys.exit(f"[FAIL] Missing package '{e.name}'.\n"
             f"       Fix: pip install mediapipe scikit-learn opencv-python joblib numpy matplotlib seaborn")

from utils.landmark_data import load_landmark_data, augment, to_features, RESULTS_DIR
from utils.landmark_models import KMeansVoteClassifier
from utils.metrics import print_metrics, plot_confusion_matrix

MODEL_PATH = RESULTS_DIR / "kmeans_model.joblib"
CLUSTER_COUNTS = [36, 108, 216]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 60 + "\n  Landmarks + K-Means  -  ASL Classifier\n" + "=" * 60)
    X_tr, y_tr, X_te, y_te, classes = load_landmark_data(args.limit, args.rebuild)

    print(f"\n[2] Trying cluster counts {CLUSTER_COUNTS} on a 20% validation split ...")
    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
    Xa_f, ya_f = augment(Xa, ya)
    Xa_f, Xv_f = to_features(Xa_f), to_features(Xv)
    best_k, best_acc = None, -1
    for k in CLUSTER_COUNTS:
        t0 = time.time()
        acc = KMeansVoteClassifier(n_clusters=k).fit(Xa_f, ya_f).score(Xv_f, yv)
        print(f"       K={k:<4} val accuracy = {acc*100:.2f}%  ({time.time()-t0:.0f}s)")
        if acc > best_acc:
            best_k, best_acc = k, acc
    print(f"\n  Best K = {best_k}")

    print("\n[3] Final fit on the full training set ...")
    X_full, y_full = augment(X_tr, y_tr)
    final = KMeansVoteClassifier(n_clusters=best_k).fit(to_features(X_full), y_full)
    joblib.dump({"pipeline": final, "classes": classes, "name": "K-Means"}, MODEL_PATH)
    print(f"       saved {MODEL_PATH}")

    print("\n[4] Test-set evaluation ...\n")
    y_pred = final.predict(to_features(X_te))
    res = print_metrics(y_te, y_pred, classes, f"Landmarks + K-Means (K={best_k})")
    plot_confusion_matrix(res["confusion_matrix"], classes, f"Landmarks + K-Means (K={best_k})",
                          save_path=str(RESULTS_DIR / "kmeans_confusion_matrix.png"))
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
