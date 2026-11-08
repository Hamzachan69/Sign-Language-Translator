"""
==============================================================================
  ASL Classifier - MediaPipe LANDMARKS + MLP (small neural network)
==============================================================================
HOW TO RUN
----------
  pip install mediapipe scikit-learn opencv-python joblib numpy matplotlib seaborn
  python models/mlp_classifier.py              # full run
  python models/mlp_classifier.py --limit 200  # quick test (200 images/class)
  python models/mlp_classifier.py --rebuild    # ignore cached landmarks

PIPELINE
--------
  image -> MediaPipe 21 landmarks -> normalise (wrist origin, scale) -> 93 features
        -> mirror + noise augmentation -> StandardScaler -> MLP(256,128) -> 36 classes

OUTPUT
------
  models/results/mlp_model.joblib   {"pipeline", "classes", "name"}
  models/results/mlp_confusion_matrix.png

UNINSTALL / CLEAN UP
--------------------
  delete models/results/mlp_* ; (shared cache) models/results/landmark_features_*.npz
"""
import sys, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    sys.exit(f"[FAIL] Missing package '{e.name}'.\n"
             f"       Fix: pip install mediapipe scikit-learn opencv-python joblib numpy matplotlib seaborn")

from utils.landmark_data import load_landmark_data, augment, to_features, RESULTS_DIR
from utils.metrics import print_metrics, plot_confusion_matrix

MODEL_PATH = RESULTS_DIR / "mlp_model.joblib"


def make_mlp():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3,
                              early_stopping=True, validation_fraction=0.1,
                              n_iter_no_change=15, max_iter=400, random_state=42)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 60 + "\n  Landmarks + MLP  -  ASL Classifier\n" + "=" * 60)
    X_tr, y_tr, X_te, y_te, classes = load_landmark_data(args.limit, args.rebuild)

    print("\n[2] Validation check (20% held out from train) ...")
    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
    Xa_aug, ya_aug = augment(Xa, ya, jitter=0.02)
    t0 = time.time()
    val_pipe = make_mlp().fit(to_features(Xa_aug), ya_aug)
    print(f"       val accuracy = {val_pipe.score(to_features(Xv), yv)*100:.2f}%  ({time.time()-t0:.0f}s)")

    print("\n[3] Final fit on the full training set ...")
    X_full, y_full = augment(X_tr, y_tr, jitter=0.02)
    final = make_mlp().fit(to_features(X_full), y_full)
    joblib.dump({"pipeline": final, "classes": classes, "name": "MLP"}, MODEL_PATH)
    print(f"       saved {MODEL_PATH}")

    print("\n[4] Test-set evaluation ...\n")
    y_pred = final.predict(to_features(X_te))
    res = print_metrics(y_te, y_pred, classes, "Landmarks + MLP")
    plot_confusion_matrix(res["confusion_matrix"], classes, "Landmarks + MLP",
                          save_path=str(RESULTS_DIR / "mlp_confusion_matrix.png"))
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
