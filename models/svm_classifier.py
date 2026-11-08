"""
==============================================================================
  ASL Classifier - MediaPipe LANDMARKS + SVM (RBF kernel)
==============================================================================
HOW TO RUN
----------
  pip install mediapipe scikit-learn opencv-python joblib numpy matplotlib seaborn
  python models/svm_classifier.py              # full run
  python models/svm_classifier.py --limit 200  # quick test
  python models/svm_classifier.py --rebuild    # ignore cached landmarks

PIPELINE
--------
  image -> MediaPipe 21 landmarks -> normalise -> 93 features -> mirror augmentation
        -> StandardScaler -> SVC(RBF) with C tuned on a validation split

OUTPUT
------
  models/results/svm_model.joblib   {"pipeline", "classes", "name"}
  models/results/svm_confusion_matrix.png

UNINSTALL / CLEAN UP
--------------------
  delete models/results/svm_* ; (shared cache) models/results/landmark_features_*.npz
"""
import sys, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as e:
    sys.exit(f"[FAIL] Missing package '{e.name}'.\n"
             f"       Fix: pip install mediapipe scikit-learn opencv-python joblib numpy matplotlib seaborn")

from utils.landmark_data import load_landmark_data, augment, to_features, RESULTS_DIR
from utils.metrics import print_metrics, plot_confusion_matrix

MODEL_PATH = RESULTS_DIR / "svm_model.joblib"
C_VALUES = [3, 10, 30]        # margin strictness candidates
TUNE_CAP = 12000              # SVM training is O(n^2): tune on a subset
FINAL_CAP = 30000             # cap for the final fit so it finishes in minutes


def make_svm(C, probability=False):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=C, kernel="rbf", gamma="scale", probability=probability, random_state=42)),
    ])


def subsample(X, y, cap, rng):
    if len(y) <= cap:
        return X, y
    sel = rng.choice(len(y), cap, replace=False)
    return X[sel], y[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 60 + "\n  Landmarks + SVM  -  ASL Classifier\n" + "=" * 60)
    X_tr, y_tr, X_te, y_te, classes = load_landmark_data(args.limit, args.rebuild)
    rng = np.random.default_rng(42)

    print(f"\n[2] Tuning C in {C_VALUES} on a 20% validation split ...")
    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
    Xa_aug, ya_aug = augment(Xa, ya)
    Xt, yt = subsample(to_features(Xa_aug), ya_aug, TUNE_CAP, rng)
    Xv_f = to_features(Xv)
    best_C, best_acc = None, -1
    for C in C_VALUES:
        t0 = time.time()
        acc = make_svm(C).fit(Xt, yt).score(Xv_f, yv)
        print(f"       C={C:<3} val accuracy = {acc*100:.2f}%  ({time.time()-t0:.0f}s)")
        if acc > best_acc:
            best_C, best_acc = C, acc
    print(f"\n  Best C = {best_C}")

    print("\n[3] Final fit on the full training set (with probabilities) ...")
    X_full, y_full = augment(X_tr, y_tr)
    Xf, yf = subsample(to_features(X_full), y_full, FINAL_CAP, rng)
    t0 = time.time()
    final = make_svm(best_C, probability=True).fit(Xf, yf)
    print(f"       fitted on {len(yf)} samples in {time.time()-t0:.0f}s")
    joblib.dump({"pipeline": final, "classes": classes, "name": "SVM"}, MODEL_PATH)
    print(f"       saved {MODEL_PATH}")

    print("\n[4] Test-set evaluation ...\n")
    y_pred = final.predict(to_features(X_te))
    res = print_metrics(y_te, y_pred, classes, "Landmarks + SVM")
    plot_confusion_matrix(res["confusion_matrix"], classes, "Landmarks + SVM",
                          save_path=str(RESULTS_DIR / "svm_confusion_matrix.png"))
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
