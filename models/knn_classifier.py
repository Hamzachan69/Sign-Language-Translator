"""
══════════════════════════════════════════════════════════════════════════════
  ASL Hand-Gesture Classifier  –  K-Nearest Neighbours (KNN)
══════════════════════════════════════════════════════════════════════════════

HOW TO RUN
----------
  python models/knn_classifier.py

WHAT IT DOES
------------
  1. Loads the pre-split asl_processed  train / test images.
  2. Flattens each image to a 1-D feature vector (64 × 64 × 3 = 12 288 dims).
  3. Applies PCA for dimensionality reduction (speeds up k-NN significantly).
  4. Performs a small grid-search over k to pick the best neighbour count.
  5. Fits the best StandardScaler -> PCA(150) -> KNeighborsClassifier pipeline.
  6. Saves the fitted pipeline to models/results/knn_model.joblib.
  7. Evaluates on the held-out test set and prints:
       • Accuracy  • Precision  • Recall  • F1-Score  • Confusion Matrix
  8. Saves the confusion-matrix heat-map as a PNG.

OUTPUT FILES
------------
  models/results/knn_model.joblib
  models/results/knn_confusion_matrix.png
"""

import os, sys, time
import joblib
import numpy as np
from pathlib import Path

# ── make sibling imports work when script is run directly ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.data_loader import TRAIN_DIR, TEST_DIR, load_split
from utils.metrics import print_metrics, plot_confusion_matrix

# ─── config ───────────────────────────────────────────────────────────────────
K_VALUES       = [3, 5, 7, 9]      # neighbour counts to try
PCA_COMPONENTS = 150               # keep 150 principal components
RESULTS_DIR    = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_PATH     = RESULTS_DIR / "knn_model.joblib"


def main():
    print("\n" + "=" * 60)
    print("  K-Nearest Neighbours  -  ASL Classifier")
    print("=" * 60)

    # ── 1. load data ──────────────────────────────────────────────────────────
    print("\n[1/4]  Loading dataset …")
    X_train, y_train, classes = load_split(TRAIN_DIR, flatten=True)
    X_test,  y_test,  _       = load_split(TEST_DIR,  flatten=True)
    print(f"       Train : {X_train.shape}   Test : {X_test.shape}")
    print(f"       Classes ({len(classes)}): {', '.join(classes)}")

    # ── 2. build pipeline (scale → PCA → KNN) ────────────────────────────────
    print(f"\n[2/4]  Building pipeline  "
          f"(StandardScaler → PCA({PCA_COMPONENTS}) → KNN) …")

    best_k, best_acc, best_pipe = None, -1, None

    print(f"\n[3/4]  Grid-searching k ∈ {K_VALUES} …")
    # Simple hold-out validation on 20 % of training data for speed
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train))
    val_size = int(0.2 * len(X_train))
    val_idx,  tr_idx = idx[:val_size], idx[val_size:]
    X_tr, y_tr  = X_train[tr_idx],  y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    for k in K_VALUES:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=PCA_COMPONENTS, random_state=42)),
            ("knn",    KNeighborsClassifier(n_neighbors=k,
                                            metric="euclidean",
                                            n_jobs=-1)),
        ])
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        val_acc = pipe.score(X_val, y_val)
        print(f"       k={k:>2d}  val-accuracy = {val_acc*100:.2f} %"
              f"  ({time.time()-t0:.1f}s)")
        if val_acc > best_acc:
            best_acc, best_k, best_pipe = val_acc, k, pipe

    print(f"\n  Best k = {best_k}  (val-accuracy = {best_acc*100:.2f} %)")

    # ── 3. final fit on full training set ─────────────────────────────────────
    print(f"\n       Re-fitting on full training set with k={best_k} …")
    best_pipe.set_params(knn__n_neighbors=best_k)
    best_pipe.fit(X_train, y_train)

    # ── 4. persist pipeline to disk ───────────────────────────────────────────
    print(f"\n       Saving fitted KNN pipeline to {MODEL_PATH} …")
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"       Model saved successfully.")

    # ── 5. evaluate ───────────────────────────────────────────────────────────
    print("\n[4/4]  Evaluating on test set …\n")
    y_pred  = best_pipe.predict(X_test)
    results = print_metrics(y_test, y_pred, classes, f"KNN (k={best_k})")

    cm_path = str(RESULTS_DIR / "knn_confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], classes,
                          f"KNN (k={best_k})", save_path=cm_path)

    print(f"\n  Done.  Results saved to: {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
