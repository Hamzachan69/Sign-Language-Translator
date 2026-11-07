"""
══════════════════════════════════════════════════════════════════════════════
  ASL Hand-Gesture Classifier  –  K-Means Clustering
══════════════════════════════════════════════════════════════════════════════

NOTE ON K-MEANS
---------------
K-Means is an **unsupervised** algorithm; it has no notion of class labels
during clustering.  After clustering, we assign each cluster a label via
majority-vote (the most common true label among its members), then treat
those assignments as predictions to compute the standard supervised metrics.

HOW TO RUN
----------
  python models/kmeans_classifier.py

WHAT IT DOES
------------
  1. Loads train / test images and flattens them.
  2. Applies StandardScaler -> PCA(150) -> MiniBatchKMeans in a scikit-learn Pipeline.
  3. Fits Mini-Batch K-Means with K = number of ASL classes (36).
  4. Maps clusters -> class labels via majority vote on the training set.
  5. Saves the fitted pipeline and cluster mapping to models/results/kmeans_model.joblib.
  6. Predicts test-set labels and prints:
       • Accuracy  • Precision  • Recall  • F1-Score  • Confusion Matrix
  7. Saves the confusion-matrix heat-map as a PNG.

OUTPUT FILES
------------
  models/results/kmeans_model.joblib
  models/results/kmeans_confusion_matrix.png
"""

import sys, time
import joblib
import numpy as np
from pathlib import Path

# ── make sibling imports work when script is run directly ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.data_loader import TRAIN_DIR, TEST_DIR, load_split
from utils.metrics import print_metrics, plot_confusion_matrix

# ─── config ───────────────────────────────────────────────────────────────────
PCA_COMPONENTS = 150          # principal components to keep (StandardScaler -> PCA(150) -> KMeans)
RANDOM_STATE   = 42
RESULTS_DIR    = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_PATH     = RESULTS_DIR / "kmeans_model.joblib"


def majority_vote_map(cluster_labels: np.ndarray,
                      true_labels: np.ndarray,
                      n_clusters: int,
                      n_classes: int) -> np.ndarray:
    """
    For each cluster ID, pick the most common true class label.
    Returns a mapping array: cluster_id  →  class_id.
    """
    mapping = np.zeros(n_clusters, dtype=np.int32)
    for c in range(n_clusters):
        mask = (cluster_labels == c)
        if mask.sum() == 0:
            continue
        # find the majority true label in this cluster
        counts = np.bincount(true_labels[mask], minlength=n_classes)
        mapping[c] = int(np.argmax(counts))
    return mapping


def main():
    print("\n" + "=" * 60)
    print("  K-Means Clustering  -  ASL Classifier")
    print("=" * 60)

    # ── 1. load data ──────────────────────────────────────────────────────────
    print("\n[1/4]  Loading dataset …")
    X_train, y_train, classes = load_split(TRAIN_DIR, flatten=True)
    X_test,  y_test,  _       = load_split(TEST_DIR,  flatten=True)
    n_classes = len(classes)
    print(f"       Train : {X_train.shape}   Test : {X_test.shape}")
    print(f"       Classes ({n_classes}): {', '.join(classes)}")

    # ── 2. build pipeline (StandardScaler → PCA(150) → MiniBatchKMeans) ──────
    print(f"\n[2/4]  Building pipeline  (StandardScaler → PCA({PCA_COMPONENTS}) → MiniBatchKMeans(K={n_classes})) …")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)),
        ("kmeans", MiniBatchKMeans(
            n_clusters=n_classes,
            random_state=RANDOM_STATE,
            n_init=10,
            max_iter=300,
            batch_size=1024,
            verbose=0,
        )),
    ])

    # ── 3. fit K-Means ────────────────────────────────────────────────────────
    print(f"\n[3/4]  Fitting MiniBatchKMeans pipeline (K={n_classes}) …")
    t0 = time.time()
    pipe.fit(X_train)
    km = pipe.named_steps["kmeans"]
    pca = pipe.named_steps["pca"]
    print(f"       Clustering done in {time.time()-t0:.1f}s  "
          f"| inertia = {km.inertia_:.0f}")
    print(f"       Explained variance retained : "
          f"{pca.explained_variance_ratio_.sum()*100:.1f} %")

    # ── 4. build cluster → label mapping via majority vote ────────────────────
    train_cluster_ids = km.labels_
    cluster_map = majority_vote_map(train_cluster_ids, y_train,
                                    n_classes, n_classes)
    pipe.cluster_map_ = cluster_map

    # ── 5. persist pipeline to disk ───────────────────────────────────────────
    print(f"\n       Saving fitted K-Means pipeline to {MODEL_PATH} …")
    joblib.dump(pipe, MODEL_PATH)
    print(f"       Model saved successfully.")

    # ── 6. predict on test set ────────────────────────────────────────────────
    print("\n[4/4]  Evaluating on test set …\n")
    test_cluster_ids = pipe.predict(X_test)
    y_pred = cluster_map[test_cluster_ids]

    results = print_metrics(y_test, y_pred, classes, "K-Means")

    cm_path = str(RESULTS_DIR / "kmeans_confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], classes,
                          "K-Means", save_path=cm_path)

    print(f"\n  Done.  Results saved to: {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
