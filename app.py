"""
=============================================================================
  ASL Sign Language Translator - Web Application (Flask)
=============================================================================

HOW TO RUN
----------
  python app.py          then open http://localhost:5000

MODELS (selectable from the UI, value sent as "model")
  landmark : MediaPipe landmarks -> MLP/RF/SVM  (CPU only, fastest, recommended)
  cnn      : Keras CNN on 64x64 hand crops
  knn      : PCA + KNN on 64x64 hand crops
  kmeans   : PCA + MiniBatchKMeans (unsupervised baseline)

If the requested model is not loaded the server falls back to another one,
and now REPORTS it ("fallback": true, "requested_model": ...) instead of
hiding it.
"""

import os
import sys
import base64
import time
import numpy as np
import cv2
import joblib
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import mediapipe as mp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "models"))
try:
    from utils.landmark_features import from_mediapipe
except ImportError as e:
    sys.exit(f"[FAIL] Cannot import models/utils/landmark_features.py ({e}). "
             "Copy that file into models/utils/ (it is shared with landmark_classifier.py).")

RESULTS_DIR       = PROJECT_ROOT / "models" / "results"
CNN_MODEL_PATH    = RESULTS_DIR / "cnn_best_model.keras"
KNN_MODEL_PATH    = RESULTS_DIR / "knn_model.joblib"
KMEANS_MODEL_PATH = RESULTS_DIR / "kmeans_model.joblib"
LANDMARK_PATH     = RESULTS_DIR / "landmark_model.joblib"
TEMPLATE_DIR      = PROJECT_ROOT / "templates"
STATIC_DIR        = PROJECT_ROOT / "static"

IMG_SIZE = 64  # must match training config (64x64)

CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]

FALLBACK_ORDER = ["landmark", "cnn", "knn", "kmeans"]

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))

# ─── Load Multi-Model Registry ────────────────────────────────────────────────
models = {"landmark": None, "cnn": None, "knn": None, "kmeans": None}

print("\n" + "=" * 60)
print("  Initializing ASL Model Registry (Landmark, CNN, KNN, K-Means)")
print("=" * 60)


def _load(name, path, loader):
    if not path.exists():
        print(f"  [-] {name} model NOT FOUND at {path}")
        return
    try:
        models[name] = loader(str(path))
        print(f"  [OK] {name} loaded: {path.name}")
    except Exception as e:
        print(f"  [!] {name} FAILED to load ({type(e).__name__}): {e}")


_load("landmark", LANDMARK_PATH, joblib.load)
_load("cnn", CNN_MODEL_PATH, tf.keras.models.load_model)
_load("knn", KNN_MODEL_PATH, joblib.load)
_load("kmeans", KMEANS_MODEL_PATH, joblib.load)

if not any(models.values()):
    print("\n  [FAIL] No model could be loaded. Train one first, e.g.:")
    print("         python models/landmark_classifier.py")

# ─── MediaPipe Hands ──────────────────────────────────────────────────────────
print("\nInitializing MediaPipe Hands ...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)
print("MediaPipe Hands initialized successfully.\n")


# ─── Helper: extract square hand ROI from frame using landmarks ───────────────
def extract_hand_roi(frame, hand_landmarks, padding=25):
    """Square crop around the hand, RGB, resized to 64x64, normalized to [0,1]."""
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    side = max(x_max - x_min, y_max - y_min) + padding * 2

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return None

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None

    pad_top, pad_bottom = y1c - y1, y2 - y2c
    pad_left, pad_right = x1c - x1, x2 - x2c
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = cv2.copyMakeBorder(
            crop,
            max(0, pad_top), max(0, pad_bottom),
            max(0, pad_left), max(0, pad_right),
            cv2.BORDER_CONSTANT, value=[0, 0, 0],
        )

    roi_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return roi_resized.astype(np.float32) / 255.0


def get_landmarks_list(hand_landmarks):
    return [
        {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
        for lm in hand_landmarks.landmark
    ]


def pick_model(requested):
    """Return (active_model, fallback_used)."""
    if requested in models and models[requested] is not None:
        return requested, False
    for name in FALLBACK_ORDER:
        if models[name] is not None:
            return name, True
    return requested, True


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("welcome.html")


@app.route("/translate")
def translate_page():
    return render_template("translator.html")


@app.route("/api/models")
def list_models():
    """Which models are actually loaded (handy for debugging / the UI)."""
    return jsonify({k: v is not None for k, v in models.items()})


@app.route("/api/predict", methods=["POST"])
def predict():
    t0 = time.time()

    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400

    requested_model = str(data.get("model", "landmark")).lower()
    active_model, fallback = pick_model(requested_model)

    try:
        img_data = data["frame"]
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {str(e)}"}), 400

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    response = {
        "hand_detected": False,
        "letter": None,
        "confidence": 0.0,
        "active_model": active_model,
        "requested_model": requested_model,
        "fallback": fallback,
        "landmarks": None,
        "handedness": None,
        "latency_ms": 0,
    }

    if result.multi_hand_landmarks:
        hand_lms = result.multi_hand_landmarks[0]
        response["hand_detected"] = True
        response["landmarks"] = get_landmarks_list(hand_lms)

        if result.multi_handedness:
            cls = result.multi_handedness[0].classification[0]
            response["handedness"] = {
                "label": "Right" if cls.label == "Left" else "Left",
                "score": round(cls.score, 3),
            }

        estimator = models.get(active_model)
        if estimator is not None:
            class_idx = None
            confidence = 0.0

            if active_model == "landmark":
                feat = from_mediapipe(hand_lms, frame.shape[1], frame.shape[0])
                if feat is not None:
                    pipe = estimator["pipeline"]
                    probs = pipe.predict_proba(feat.reshape(1, -1))[0]
                    best = int(np.argmax(probs))
                    class_idx = int(pipe.classes_[best])
                    confidence = float(probs[best])
            else:
                roi = extract_hand_roi(frame, hand_lms)
                if roi is not None:
                    if active_model == "cnn":
                        preds = estimator.predict(np.expand_dims(roi, axis=0), verbose=0)
                        class_idx = int(np.argmax(preds[0]))
                        confidence = float(preds[0][class_idx])

                    elif active_model == "knn":
                        flat = roi.reshape(1, -1)
                        probs = estimator.predict_proba(flat)[0]
                        class_idx = int(estimator.classes_[int(np.argmax(probs))])
                        confidence = float(np.max(probs))

                    elif active_model == "kmeans":
                        flat = roi.reshape(1, -1)
                        cluster_id = int(estimator.predict(flat)[0])
                        if hasattr(estimator, "cluster_map_"):
                            class_idx = int(estimator.cluster_map_[cluster_id])
                        else:
                            class_idx = cluster_id % len(CLASSES)
                        try:
                            dists = estimator.transform(flat)[0]
                            exp_neg = np.exp(-dists + np.min(dists))
                            confidence = float(exp_neg[cluster_id] / np.sum(exp_neg))
                        except Exception:
                            confidence = 0.5

            if class_idx is not None and 0 <= class_idx < len(CLASSES):
                response["letter"] = CLASSES[class_idx]
                response["confidence"] = round(confidence, 4)

    response["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonify(response)


if __name__ == "__main__":
    print("=" * 60)
    print("  ASL Sign Language Translator")
    print("  Open http://localhost:5000 in your browser")
    print("  Loaded models: " + ", ".join(k for k, v in models.items() if v is not None))
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
