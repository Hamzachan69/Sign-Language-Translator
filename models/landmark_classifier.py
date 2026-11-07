"""
==============================================================================
  ASL Classifier - MediaPipe LANDMARKS + (MLP | Random Forest | SVM)
==============================================================================
Instead of feeding raw pixels (background / lighting / skin-tone sensitive),
we classify the 21 hand landmarks (63 numbers) that MediaPipe already gives us.

HOW TO RUN
----------
  pip install mediapipe scikit-learn opencv-python joblib
  python models/landmark_classifier.py              # full run (feature extraction ~10-20 min once, then cached)
  python models/landmark_classifier.py --limit 200  # quick test: 200 images per class
  python models/landmark_classifier.py --rebuild    # ignore cached features

OUTPUT
------
  models/results/landmark_features_{train,test}.npz   (cache)
  models/results/landmark_model.joblib   {"pipeline": ..., "classes": [...], "name": ...}
  models/results/landmark_confusion_matrix.png

UNINSTALL / CLEAN UP
--------------------
  delete models/results/landmark_* and models/utils/landmark_features.py
"""
import sys, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2
    import joblib
    import numpy as np
    import mediapipe as mp
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as e:
    sys.exit(f"[FAIL] Missing package: {e.name}. Fix: pip install mediapipe scikit-learn opencv-python joblib numpy")

from utils.data_loader import TRAIN_DIR, TEST_DIR
from utils.metrics import print_metrics, plot_confusion_matrix
from utils.landmark_features import from_mediapipe, mirror

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = RESULTS_DIR / "landmark_model.joblib"
EXTS = {".jpg", ".jpeg", ".png"}


def extract_split(split_dir, cache_path, limit, rebuild):
    split_dir = Path(split_dir)
    if cache_path.exists() and not rebuild and not limit:
        d = np.load(cache_path, allow_pickle=True)
        print(f"       cache hit: {cache_path.name}  ({len(d['y'])} samples)")
        return d["X"], d["y"], list(d["classes"])

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
    if not class_dirs:
        sys.exit(f"[FAIL] No class folders in {split_dir}. Check TRAIN_DIR/TEST_DIR in utils/data_loader.py")
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
                # pad: tightly cropped dataset images often defeat the hand detector
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
            print(f"       [{ci+1:>2}/{len(class_dirs)}] {cdir.name:<3} detected {len(files)-miss}/{len(files)}"
                  f"   ({time.time()-t0:.0f}s)")

    if not X:
        sys.exit("[FAIL] MediaPipe found no hands in any image. Open a few dataset images and check "
                 "they actually show a hand; try lowering min_detection_confidence.")
    bad = {k: v for k, v in missed.items() if v[1] and v[0] / v[1] > 0.3}
    if bad:
        print(f"  [WARN] >30% detection misses in classes: {bad}  (these classes will be weak)")

    X, y = np.asarray(X, np.float32), np.asarray(y, np.int32)
    if not limit:
        np.savez(cache_path, X=X, y=y, classes=np.array(classes))
    return X, y, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max images per class (quick test)")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    print("\n[1/4] Extracting MediaPipe landmarks ...")
    print("     train:")
    X_tr, y_tr, classes = extract_split(TRAIN_DIR, RESULTS_DIR / "landmark_features_train.npz", args.limit, args.rebuild)
    print("     test:")
    X_te, y_te, classes_te = extract_split(TEST_DIR, RESULTS_DIR / "landmark_features_test.npz", args.limit, args.rebuild)
    if classes != classes_te:
        sys.exit("[FAIL] train and test class folders differ.")
    print(f"       train {X_tr.shape}  test {X_te.shape}  classes {len(classes)}")

    # mirrored copies -> works for left and right hands
    def with_mirror(X, y):
        return np.vstack([X, np.apply_along_axis(mirror, 1, X)]), np.concatenate([y, y])

    print("\n[2/4] Model selection on a 20% validation split ...")
    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
    Xa_m, ya_m = with_mirror(Xa, ya)

    candidates = {
        "MLP": (lambda: MLPClassifier((256, 128), early_stopping=True, max_iter=300, random_state=42), None),
        "RandomForest": (lambda: RandomForestClassifier(300, n_jobs=-1, random_state=42), None),
        # SVC is O(n^2): cap its training size so it finishes in minutes
        "SVM-RBF": (lambda: SVC(C=10, gamma="scale", probability=True, random_state=42), 20000),
    }
    best_name, best_acc = None, -1
    rng = np.random.default_rng(42)
    for name, (make, cap) in candidates.items():
        Xf, yf = Xa_m, ya_m
        if cap and len(yf) > cap:
            sel = rng.choice(len(yf), cap, replace=False)
            Xf, yf = Xf[sel], yf[sel]
        t0 = time.time()
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", make())]).fit(Xf, yf)
        acc = pipe.score(Xv, yv)
        print(f"       {name:<13} val-acc = {acc*100:6.2f}%   ({time.time()-t0:.0f}s)")
        if acc > best_acc:
            best_name, best_acc = name, acc
    print(f"\n  Winner: {best_name}  ({best_acc*100:.2f}% val)")

    print("\n[3/4] Refit winner on full training set ...")
    make, cap = candidates[best_name]
    Xf, yf = with_mirror(X_tr, y_tr)
    if cap and len(yf) > cap:
        sel = rng.choice(len(yf), cap, replace=False)
        Xf, yf = Xf[sel], yf[sel]
    final = Pipeline([("scaler", StandardScaler()), ("clf", make())]).fit(Xf, yf)
    joblib.dump({"pipeline": final, "classes": classes, "name": best_name}, MODEL_PATH)
    print(f"       saved {MODEL_PATH}")

    print("\n[4/4] Test-set evaluation ...\n")
    y_pred = final.predict(X_te)
    res = print_metrics(y_te, y_pred, classes, f"Landmarks + {best_name}")
    plot_confusion_matrix(res["confusion_matrix"], classes, f"Landmarks + {best_name}",
                          save_path=str(RESULTS_DIR / "landmark_confusion_matrix.png"))
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
