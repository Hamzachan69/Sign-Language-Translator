"""
Shared landmark dataset builder for the MLP and SVM trainers.

Runs MediaPipe over the dataset ONCE and caches the 63-number vectors:
    models/results/landmark_features_train.npz
    models/results/landmark_features_test.npz
(same cache files your earlier landmark_classifier.py wrote, so they are reused).
"""
import sys, time
from pathlib import Path

try:
    import cv2
    import numpy as np
    import mediapipe as mp
except ImportError as e:
    sys.exit(f"[FAIL] Missing package '{e.name}'.\n"
             f"       Fix: pip install mediapipe opencv-python numpy")

from utils.data_loader import TRAIN_DIR, TEST_DIR
from utils.landmark_features import from_mediapipe, mirror, featurize_many

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)
EXTS = {".jpg", ".jpeg", ".png"}


def _extract_split(split_dir, cache_path, limit, rebuild):
    split_dir = Path(split_dir)
    if cache_path.exists() and not rebuild and not limit:
        d = np.load(cache_path, allow_pickle=True)
        print(f"       cache hit: {cache_path.name}  ({len(d['y'])} samples)")
        return d["X"], d["y"], [str(c) for c in d["classes"]]

    if not split_dir.exists():
        sys.exit(f"[FAIL] Folder not found: {split_dir}\n"
                 f"       Check TRAIN_DIR / TEST_DIR in models/utils/data_loader.py")
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
    if not class_dirs:
        sys.exit(f"[FAIL] No class sub-folders in {split_dir}")
    classes = [d.name for d in class_dirs]

    X, y, missed = [], [], {}
    t0 = time.time()
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                  model_complexity=1, min_detection_confidence=0.3) as hands:
        for ci, cdir in enumerate(class_dirs):
            files = sorted(p for p in cdir.iterdir() if p.suffix.lower() in EXTS)
            if limit:
                files = files[:limit]
            miss = 0
            for f in files:
                img = cv2.imread(str(f))
                if img is None:
                    miss += 1
                    continue
                # tightly cropped images defeat the detector -> pad with the mean colour
                p = int(0.3 * max(img.shape[:2]))
                fill = [float(v) for v in img.reshape(-1, 3).mean(0)]
                img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=fill)
                res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not res.multi_hand_landmarks:
                    miss += 1
                    continue
                feat = from_mediapipe(res.multi_hand_landmarks[0], img.shape[1], img.shape[0])
                if feat is None:
                    miss += 1
                    continue
                X.append(feat)
                y.append(ci)
            missed[cdir.name] = (miss, len(files))
            print(f"       [{ci+1:>2}/{len(class_dirs)}] {cdir.name:<3} detected "
                  f"{len(files)-miss}/{len(files)}   ({time.time()-t0:.0f}s)")

    if not X:
        sys.exit("[FAIL] MediaPipe found no hands in any image. Open a few dataset images and "
                 "check they show a hand.")
    bad = {k: v for k, v in missed.items() if v[1] and v[0] / v[1] > 0.3}
    if bad:
        print(f"  [WARN] >30% detection misses in classes: {bad}  (these classes will be weak)")

    X, y = np.asarray(X, np.float32), np.asarray(y, np.int32)
    if not limit:
        np.savez(cache_path, X=X, y=y, classes=np.array(classes))
    return X, y, classes


def load_landmark_data(limit=0, rebuild=False):
    """Returns raw 63-dim train/test arrays + class names."""
    print("\n[1] Extracting MediaPipe landmarks ...")
    print("     train:")
    X_tr, y_tr, classes = _extract_split(TRAIN_DIR, RESULTS_DIR / "landmark_features_train.npz", limit, rebuild)
    print("     test:")
    X_te, y_te, classes_te = _extract_split(TEST_DIR, RESULTS_DIR / "landmark_features_test.npz", limit, rebuild)
    if classes != classes_te:
        sys.exit("[FAIL] train and test class folders differ.")
    print(f"       train {X_tr.shape}  test {X_te.shape}  classes {len(classes)}")
    return X_tr, y_tr, X_te, y_te, classes


def augment(X_raw, y, jitter=0.0, seed=42):
    """Mirror every sample (left+right hands). Optionally add a noisy copy of everything.

    Mirroring happens on the raw 63 numbers BEFORE featurize(), so engineered
    features stay consistent.
    """
    Xm = np.apply_along_axis(mirror, 1, X_raw)
    Xa, ya = np.vstack([X_raw, Xm]), np.concatenate([y, y])
    if jitter > 0:
        rng = np.random.default_rng(seed)
        Xn = Xa + rng.normal(0, jitter, Xa.shape).astype(np.float32)
        Xa, ya = np.vstack([Xa, Xn]), np.concatenate([ya, ya])
    return Xa, ya


def to_features(X_raw):
    return featurize_many(X_raw)
