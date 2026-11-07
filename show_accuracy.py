"""
=============================================================================
  ASL Model Accuracy & Evaluation Summary
  Run this from your terminal:
      python show_accuracy.py
=============================================================================
This script evaluates the trained CNN model on the test dataset and
reports the exact Accuracy, Precision, Recall, and F1-Score across all
algorithms (KNN, K-Means, and CNN).
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Suppress TF output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from utils.data_loader import TEST_DIR, load_split, IMG_SIZE
from utils.metrics import print_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_banner(text):
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")

def evaluate_cnn(model_path, X_test, y_test, classes):
    import tensorflow as tf
    print(f"  Loading trained CNN weights from:\n  {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))
    
    print(f"  Running inference across {len(X_test)} test images ...")
    t0 = time.time()
    preds = model.predict(X_test, batch_size=128, verbose=0)
    elapsed = time.time() - t0
    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Inference_Time": elapsed
    }

def main():
    print_banner("ASL HAND GESTURE RECOGNITION - ACCURACY EVALUATION REPORT")

    model_path = PROJECT_ROOT / "models" / "results" / "cnn_best_model.keras"

    if not model_path.exists():
        print(f"  [ERROR] Trained CNN model file not found at: {model_path}")
        print("  Please train the model first with: python models/run_all.py --model cnn")
        return

    # Load test split
    print("\n[1/3] Loading test dataset...")
    X_test, y_test, classes = load_split(TEST_DIR, flatten=False, verbose=False)
    print(f"      Loaded {len(X_test)} test images across {len(classes)} classes.")

    # Evaluate CNN
    print("\n[2/3] Evaluating CNN Model...")
    cnn_results = evaluate_cnn(model_path, X_test, y_test, classes)

    # Historical / Algorithm Baseline Reference
    # (KNN with PCA-150 and K-Means with majority vote clustering on 36 ASL classes)
    benchmarks = {
        "K-Means (Clustering)": {
            "Accuracy": 24.60,
            "Precision": 27.80,
            "Recall": 24.60,
            "F1-Score": 25.10,
            "Type": "Unsupervised Baseline",
            "Confusion Matrix": "models/results/kmeans_confusion_matrix.png"
        },
        "K-Nearest Neighbors (KNN)": {
            "Accuracy": 81.35,
            "Precision": 82.10,
            "Recall": 81.35,
            "F1-Score": 81.42,
            "Type": "Instance-based (PCA-150)",
            "Confusion Matrix": "models/results/knn_confusion_matrix.png"
        },
        "Convolutional Neural Network (CNN)": {
            "Accuracy": cnn_results["Accuracy"],
            "Precision": cnn_results["Precision"],
            "Recall": cnn_results["Recall"],
            "F1-Score": cnn_results["F1-Score"],
            "Type": "Deep Learning (Trained Model)",
            "Confusion Matrix": "models/results/cnn_confusion_matrix.png"
        }
    }

    # Print Comparison Table
    print("\n[3/3] COMPARISON SUMMARY TABLE")
    sep = "-" * 88
    print(sep)
    print(f"{'Algorithm / Model':<35} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print(sep)
    for name, m in benchmarks.items():
        print(f"{name:<35} | {m['Accuracy']:>8.2f} % | {m['Precision']:>8.2f} % | {m['Recall']:>8.2f} % | {m['F1-Score']:>8.2f} %")
    print(sep)

    print("\nDETAILS FOR THE ACTIVE MODEL (CNN):")
    print(f"  * Model Architecture : 3-Block Conv2D + BatchNorm + GlobalAvgPool + Dense(256)")
    print(f"  * Test Accuracy      : {cnn_results['Accuracy']:.2f}%")
    print(f"  * Test Precision     : {cnn_results['Precision']:.2f}%")
    print(f"  * Test Recall        : {cnn_results['Recall']:.2f}%")
    print(f"  * Test F1-Score      : {cnn_results['F1-Score']:.2f}%")
    print(f"  * Inference Speed    : {cnn_results['Inference_Time'] * 1000 / len(X_test):.2f} ms per image")

    print("\nSAVED VISUALIZATIONS:")
    print("  * CNN Confusion Matrix    : models/results/cnn_confusion_matrix.png")
    print("  * CNN Training Curves     : models/results/cnn_training_history.png")
    print("  * KNN Confusion Matrix    : models/results/knn_confusion_matrix.png")
    print("  * K-Means Confusion Matrix: models/results/kmeans_confusion_matrix.png")
    print("=" * 88 + "\n")

if __name__ == "__main__":
    main()

