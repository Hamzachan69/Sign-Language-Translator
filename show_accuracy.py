"""
Evaluate all trained landmark models (KNN, K-Means, SVM, MLP, 1D-CNN) on the cached test landmarks.

  python show_accuracy.py

Needs: models/results/landmark_features_test.npz (created by the trainers)
       models/results/mlp_model.joblib / svm_model.joblib
UNINSTALL: delete this file.
"""
import sys, time
from pathlib import Path

try:
    import numpy as np
    import joblib
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except ImportError as e:
    sys.exit(f"[FAIL] Missing package '{e.name}'. Fix: pip install scikit-learn joblib numpy")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models"))
from utils.landmark_features import featurize_many

R = ROOT / "models" / "results"
cache = R / "landmark_features_test.npz"
if not cache.exists():
    sys.exit(f"[FAIL] {cache} missing. Train first: python models/run_all.py")

d = np.load(cache, allow_pickle=True)
X = featurize_many(d["X"])
y = d["y"]

print("=" * 82)
print(f"  {'Model':<22}| {'Accuracy':>9} | {'Precision':>9} | {'Recall':>9} | {'F1':>9} | ms/sample")
print("=" * 82)
found = False
for name in ("knn", "kmeans", "svm", "mlp", "cnn"):
    t0 = time.time()
    if name == "cnn":
        p = R / "cnn_landmark.keras"
        if not p.exists():
            print(f"  {'CNN':<22}| not trained  (python models/cnn_classifier.py)")
            continue
        try:
            import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf
        except ImportError:
            print(f"  {'CNN':<22}| tensorflow missing  (pip install tensorflow==2.15.0)")
            continue
        model = tf.keras.models.load_model(str(p))
        t0 = time.time()
        pred = np.argmax(model.predict(d["X"].reshape(-1, 21, 3), batch_size=512, verbose=0), axis=1)
    else:
        p = R / f"{name}_model.joblib"
        if not p.exists():
            print(f"  {name.upper():<22}| not trained  (python models/{name}_classifier.py)")
            continue
        pipe = joblib.load(p)["pipeline"]
        t0 = time.time()
        pred = pipe.predict(X)
    found = True
    ms = (time.time() - t0) * 1000 / len(y)
    print(f"  Landmarks + {name.upper():<10}| "
          f"{accuracy_score(y, pred)*100:8.2f}% | "
          f"{precision_score(y, pred, average='weighted', zero_division=0)*100:8.2f}% | "
          f"{recall_score(y, pred, average='weighted', zero_division=0)*100:8.2f}% | "
          f"{f1_score(y, pred, average='weighted', zero_division=0)*100:8.2f}% | {ms:.3f}")
print("=" * 82)
if not found:
    sys.exit("[FAIL] No trained models found.")
