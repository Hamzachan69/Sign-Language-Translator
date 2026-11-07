"""
=============================================================================
  ASL Sign Language Translator - Web Application (Flask + WebSocket)
=============================================================================

HOW TO RUN
----------
  python app.py

  Then open http://localhost:5000 in your browser.
  Click "Start Now" -> camera turns on -> live ASL recognition.

ARCHITECTURE
------------
  Browser (JS)              Flask Server (Python)
  +-----------------+       +---------------------------------------------+
  | Webcam capture  |       | MediaPipe hand tracking                     |
  | via getUserMedia|------>| Multi-Model Inference (CNN / KNN / K-Means) |
  | Canvas to JPEG  | HTTP  | Return: letter, conf, active model,         |
  | frames ~12-15fps|<------| landmarks, handedness                       |
  +-----------------+       +---------------------------------------------+
"""

import os
import sys
import json
import base64
import time
import numpy as np
import cv2
import joblib
from pathlib import Path
from io import BytesIO

# ── Flask ──
from flask import Flask, render_template, request, jsonify, Response

# ── MediaPipe ──
import mediapipe as mp

# ── TensorFlow (suppress logs) ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parent
RESULTS_DIR       = PROJECT_ROOT / "models" / "results"
CNN_MODEL_PATH    = RESULTS_DIR / "cnn_best_model.keras"
KNN_MODEL_PATH    = RESULTS_DIR / "knn_model.joblib"
KMEANS_MODEL_PATH = RESULTS_DIR / "kmeans_model.joblib"
TEMPLATE_DIR      = PROJECT_ROOT / "templates"
STATIC_DIR        = PROJECT_ROOT / "static"

IMG_SIZE = 64  # must match training config (64x64)

# 36 classes in sorted order (same as training data_loader)
CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)

# ─── Load Multi-Model Registry ────────────────────────────────────────────────
models = {
    "cnn": None,
    "knn": None,
    "kmeans": None,
}

print("\n" + "=" * 60)
print("  Initializing ASL Model Registry (CNN, KNN, K-Means)")
print("=" * 60)

# 1. Load CNN model
if CNN_MODEL_PATH.exists():
    try:
        models["cnn"] = tf.keras.models.load_model(str(CNN_MODEL_PATH))
        print(f"  [✓] CNN loaded: {CNN_MODEL_PATH.name} (input shape: {models['cnn'].input_shape})")
    except Exception as e:
        print(f"  [!] Failed to load CNN model: {e}")
else:
    print(f"  [-] CNN model not found at {CNN_MODEL_PATH}")

# 2. Load KNN model
if KNN_MODEL_PATH.exists():
    try:
        models["knn"] = joblib.load(str(KNN_MODEL_PATH))
        print(f"  [✓] KNN loaded: {KNN_MODEL_PATH.name}")
    except Exception as e:
        print(f"  [!] Failed to load KNN model: {e}")
else:
    print(f"  [-] KNN model not found at {KNN_MODEL_PATH}")

# 3. Load K-Means model
if KMEANS_MODEL_PATH.exists():
    try:
        models["kmeans"] = joblib.load(str(KMEANS_MODEL_PATH))
        print(f"  [✓] K-Means loaded: {KMEANS_MODEL_PATH.name}")
    except Exception as e:
        print(f"  [!] Failed to load K-Means model: {e}")
else:
    print(f"  [-] K-Means model not found at {KMEANS_MODEL_PATH}")

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
    """
    Crop the hand region from the frame using MediaPipe landmarks.
    Pads the bounding box to a square (1:1 aspect ratio) centered on the hand,
    clamped within frame dimensions (with padding if near borders).
    Converts ROI to RGB and normalizes pixel values to [0, 1].
    """
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    box_w = x_max - x_min
    box_h = y_max - y_min
    side = max(box_w, box_h) + padding * 2

    # Target square coordinates centered on the hand
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))

    # Clamp coordinates to frame boundaries
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(w, x2)
    y2_clamped = min(h, y2)

    if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
        return None

    crop = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    if crop.size == 0:
        return None

    # If crop is not square due to boundary clamping, pad with border to restore 1:1
    pad_top = y1_clamped - y1
    pad_bottom = y2 - y2_clamped
    pad_left = x1_clamped - x1
    pad_right = x2 - x2_clamped

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = cv2.copyMakeBorder(
            crop,
            max(0, pad_top), max(0, pad_bottom),
            max(0, pad_left), max(0, pad_right),
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    # Convert BGR to RGB (Dataset was trained on RGB images; cv2 loads/decodes BGR)
    roi_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Resize to model input size (64, 64)
    roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    roi_norm = roi_resized.astype(np.float32) / 255.0
    return roi_norm


def get_landmarks_list(hand_landmarks):
    """
    Return normalized MediaPipe landmarks {x, y, z} in [0, 1] coordinate space.
    The frontend scales them directly to canvas dimensions.
    """
    return [
        {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
        for lm in hand_landmarks.landmark
    ]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the welcome page."""
    return render_template("welcome.html")


@app.route("/translate")
def translate_page():
    """Serve the live translator page."""
    return render_template("translator.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Receive a base64-encoded JPEG frame and model choice from the browser.
    Run MediaPipe hand detection + selected model classification.
    Return JSON with predicted letter, confidence, active model name, and landmarks.
    """
    t0 = time.time()

    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400

    requested_model = data.get("model", "cnn").lower()
    if requested_model not in models or models[requested_model] is None:
        # Fallback to an available model if the requested one is not loaded
        if models["cnn"] is not None:
            active_model = "cnn"
        elif models["knn"] is not None:
            active_model = "knn"
        elif models["kmeans"] is not None:
            active_model = "kmeans"
        else:
            active_model = requested_model
    else:
        active_model = requested_model

    # Decode base64 image
    try:
        img_data = data["frame"]
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {str(e)}"}), 400

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Run MediaPipe hand tracking (RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    response = {
        "hand_detected": False,
        "letter": None,
        "confidence": 0.0,
        "active_model": active_model,
        "landmarks": None,
        "handedness": None,
        "latency_ms": 0,
    }

    if result.multi_hand_landmarks:
        hand_lms = result.multi_hand_landmarks[0]
        response["hand_detected"] = True
        response["landmarks"] = get_landmarks_list(hand_lms)

        # Handedness: Invert label because raw webcam frame is viewed in selfie mode
        if result.multi_handedness:
            cls = result.multi_handedness[0].classification[0]
            corrected_label = "Right" if cls.label == "Left" else "Left"
            response["handedness"] = {
                "label": corrected_label,
                "score": round(cls.score, 3),
            }

        # Multi-model inference
        selected_estimator = models.get(active_model)
        if selected_estimator is not None:
            roi = extract_hand_roi(frame, hand_lms)
            if roi is not None:
                class_idx = None
                confidence = 0.0

                if active_model == "cnn":
                    # CNN expects (1, 64, 64, 3) tensor
                    inp = np.expand_dims(roi, axis=0)
                    preds = selected_estimator.predict(inp, verbose=0)
                    class_idx = int(np.argmax(preds[0]))
                    confidence = float(preds[0][class_idx])

                elif active_model == "knn":
                    # KNN pipeline expects flattened (1, 12288) vector
                    flat = roi.reshape(1, -1)
                    if hasattr(selected_estimator, "predict_proba"):
                        probs = selected_estimator.predict_proba(flat)[0]
                        class_idx = int(np.argmax(probs))
                        confidence = float(probs[class_idx])
                    else:
                        class_idx = int(selected_estimator.predict(flat)[0])
                        confidence = 1.0

                elif active_model == "kmeans":
                    # K-Means pipeline expects flattened (1, 12288) vector
                    flat = roi.reshape(1, -1)
                    cluster_id = int(selected_estimator.predict(flat)[0])
                    # Map cluster ID to class label via cluster_map_
                    if hasattr(selected_estimator, "cluster_map_"):
                        class_idx = int(selected_estimator.cluster_map_[cluster_id])
                    else:
                        class_idx = cluster_id % len(CLASSES)

                    # Compute soft confidence via distances to centroids
                    try:
                        dists = selected_estimator.transform(flat)[0]
                        exp_neg = np.exp(-dists + np.min(dists))
                        confidence = float(exp_neg[cluster_id] / np.sum(exp_neg))
                    except Exception:
                        confidence = 0.5

                if class_idx is not None and 0 <= class_idx < len(CLASSES):
                    response["letter"] = CLASSES[class_idx]
                    response["confidence"] = round(confidence, 4)

    elapsed = (time.time() - t0) * 1000
    response["latency_ms"] = round(elapsed, 1)

    return jsonify(response)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ASL Sign Language Translator")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
