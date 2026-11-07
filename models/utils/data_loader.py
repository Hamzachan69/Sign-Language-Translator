"""
Shared data loading utilities for all three ASL classifiers.
Loads images from the asl_processed train/test split.
"""

import os
import numpy as np
import cv2
from pathlib import Path

# ─── Dataset path ────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset" / \
    "ASL-HG American Sign Language Hand Gesture Image D" / \
    "ASL_HG_36000" / "asl_processed"

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR  = DATASET_ROOT / "test"

IMG_SIZE  = 64   # resize all images to 64×64


def load_split(split_dir: Path, img_size: int = IMG_SIZE,
               flatten: bool = True, verbose: bool = True):
    """
    Load a dataset split (train or test) from a folder of class sub-dirs.

    Parameters
    ----------
    split_dir : Path
        Root directory containing one sub-folder per class.
    img_size  : int
        Images are resized to (img_size × img_size).
    flatten   : bool
        If True  → returns (N, img_size*img_size*3) float32 array  (for KNN / KMeans)
        If False → returns (N, img_size, img_size, 3) float32 array (for CNN)
    verbose   : bool
        Print progress.

    Returns
    -------
    X : np.ndarray  float32, pixel values in [0, 1]
    y : np.ndarray  int, class indices
    classes : list[str]  sorted class names (labels)
    """
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    images, labels = [], []

    for cls in classes:
        cls_dir = split_dir / cls
        files   = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        if verbose:
            print(f"  Loading {cls:>2s}: {len(files)} images", end="\r")
        for fpath in files:
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(class_to_idx[cls])

    if verbose:
        print(f"\n  [OK] Loaded {len(images)} images from '{split_dir.name}' split")

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)

    if flatten:
        X = X.reshape(len(X), -1)   # (N, 64*64*3)

    return X, y, classes

