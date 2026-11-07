"""
Shared evaluation utilities – prints and plots all required metrics.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on all machines)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


# ─── console helpers ─────────────────────────────────────────────────────────
def _divider(char="-", width=60):
    print(char * width)


def print_metrics(y_true, y_pred, classes: list, model_name: str):
    """
    Print Accuracy, Precision, Recall, F1-Score to the console
    and return a dict with every numeric result.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    _divider("=")
    print(f"  {model_name}  -  Evaluation Results")
    _divider("=")
    print(f"  {'Accuracy':<15} : {acc * 100:.2f} %")
    print(f"  {'Precision':<15} : {prec * 100:.2f} %  (weighted)")
    print(f"  {'Recall':<15} : {rec * 100:.2f} %  (weighted)")
    print(f"  {'F1-Score':<15} : {f1 * 100:.2f} %  (weighted)")
    _divider()
    print("\n  Per-class report:\n")
    print(classification_report(y_true, y_pred,
                                target_names=classes, zero_division=0))
    _divider("=")

    return dict(accuracy=acc, precision=prec, recall=rec,
                f1=f1, confusion_matrix=cm)


# ─── confusion-matrix plot ────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, classes: list,
                          model_name: str, save_path: str = None):
    """
    Draw and optionally save an annotated confusion-matrix heat-map.
    For large class counts (> 15) annotations are hidden for readability.
    """
    n = len(classes)
    fig_size = max(12, n // 2)
    annot = n <= 15          # annotate cells only when there are few classes

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=annot, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.4 if annot else 0,
        ax=ax,
    )
    ax.set_title(f"{model_name}  –  Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Confusion matrix saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─── training-history plot (CNN only) ────────────────────────────────────────
def plot_training_history(history, model_name: str, save_path: str = None):
    """Plot training/validation accuracy & loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"{model_name} – Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history["loss"],     label="Train Loss")
    axes[1].plot(history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{model_name} – Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(model_name, fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Training history saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
