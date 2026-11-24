"""
ASL Sign Language Translator - Flask app (landmark KNN / K-Means / SVM / MLP / 1D-CNN)

  python app.py   ->  open http://localhost:5000

Browser sends a JPEG frame -> MediaPipe HandLandmarker finds 21 landmarks -> the SAME
from_mediapipe() + featurize() used in training -> selected model -> letter.

Uses the MediaPipe *Tasks* API (mediapipe >= 0.10 / 1.x removed mp.solutions).
Needs models/hand_landmarker.task (auto-downloaded on first run).

UNINSTALL: delete app.py, models/hand_landmarker.task, models/results/*.joblib, cnn_landmark.keras
"""
import os, sys, json, base64, time, urllib.request
from pathlib import Path
from types import SimpleNamespace

# Run the CNN on CPU inside the web server: it is tiny, and this stops TensorFlow
# from grabbing the whole GPU (that caused the CUDA out-of-memory messages).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Ensure required Python packages are installed before importing them
import subprocess, sys, os

def ensure_package(module_name, pip_name=None):
    """Attempt to import a module; if missing, install it via pip and import again."""
    try:
        __import__(module_name)
    except ImportError:
        package = pip_name if pip_name else module_name
        print(f"[INFO] Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        __import__(module_name)

# List of required packages (module name, pip name if different)
required_packages = [
    ("numpy", None),
    ("cv2", "opencv-python"),
    ("joblib", None),
    ("mediapipe", None),
    ("flask", None),
]
for mod, pip in required_packages:
    ensure_package(mod, pip)

# Now safe to import the modules
import numpy as np
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from flask import Flask, render_template, request, jsonify

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "models"))
from utils.landmark_features import from_mediapipe, featurize

RESULTS_DIR = PROJECT_ROOT / "models" / "results"
HAND_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_FILES = {"mlp": RESULTS_DIR / "mlp_model.joblib",
               "svm": RESULTS_DIR / "svm_model.joblib",
               "knn": RESULTS_DIR / "knn_model.joblib",
               "kmeans": RESULTS_DIR / "kmeans_model.joblib"}
CNN_PATH = RESULTS_DIR / "cnn_landmark.keras"
CNN_CLASSES = RESULTS_DIR / "cnn_classes.json"

app = Flask(__name__,
            template_folder=str(PROJECT_ROOT / "templates"),
            static_folder=str(PROJECT_ROOT / "static"))

# ─── hand landmark model (download once) ─────────────────────────────────────
if not HAND_MODEL.exists():
    print(f"Downloading hand_landmarker.task (~7.5 MB) ...")
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)
    except Exception as e:
        HAND_MODEL.unlink(missing_ok=True)
        sys.exit(f"[FAIL] Could not download the hand model: {e}\n"
                 f"       Fix manually: curl -L -o \"{HAND_MODEL}\" {HAND_MODEL_URL}")

# ─── classifier registry ─────────────────────────────────────────────────────
models = {}
print("\n" + "=" * 60 + "\n  Loading ASL landmark models\n" + "=" * 60)
for key, path in MODEL_FILES.items():
    if not path.exists():
        print(f"  [-] {key}: not found ({path.name}). Train it: python models/{key}_classifier.py")
        continue
    try:
        models[key] = joblib.load(path)
        print(f"  [OK] {key}: {path.name}")
    except Exception as e:
        print(f"  [!] {key}: failed to load: {e}")

if CNN_PATH.exists() and CNN_CLASSES.exists():
    try:
        import tensorflow as tf
        models["cnn"] = {"keras": tf.keras.models.load_model(str(CNN_PATH)),
                         "classes": json.loads(CNN_CLASSES.read_text())}
        print(f"  [OK] cnn: {CNN_PATH.name} (CPU)")
    except Exception as e:
        print(f"  [!] cnn: failed to load: {e}")
else:
    print("  [-] cnn: not found. Train it: python models/cnn_classifier.py")

if not models:
    sys.exit("[FAIL] No trained models found. Run: python models/run_all.py")

# ─── MediaPipe HandLandmarker (Tasks API, one frame at a time) ───────────────
landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
print("MediaPipe HandLandmarker ready.\n")


LABELS = {"mlp": "MLP (neural net)", "svm": "SVM (RBF)", "knn": "KNN",
          "kmeans": "K-Means", "cnn": "1D-CNN"}
ORDER = ["svm", "mlp", "cnn", "knn", "kmeans"]


def normalize_model(name):
    """'K-Means', 'k_means', '1D-CNN' ... -> kmeans / cnn ; unknown -> None."""
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    key = {"1dcnn": "cnn", "landmarkcnn": "cnn"}.get(key, key)
    return key if key in models else None


@app.route("/api/models")
def list_models():
    """The UI calls this to build its dropdown from whatever is actually trained/loaded."""
    ids = [m for m in ORDER if m in models]
    return jsonify({"models": [{"id": m, "label": LABELS.get(m, m.upper())} for m in ids],
                    "default": ids[0]})


@app.route("/")
def index():
    return render_template("welcome.html")


@app.route("/translate")
def translate_page():
    return render_template("translator.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    t0 = time.time()
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400

    active = normalize_model(data.get("model", "")) or next(m for m in ORDER if m in models)

    try:
        img = data["frame"].split(",", 1)[-1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(img), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {e}"}), 400
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    h, w = frame.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    resp = {"hand_detected": False, "letter": None, "confidence": 0.0,
            "active_model": active, "landmarks": None, "handedness": None, "latency_ms": 0}

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        resp["hand_detected"] = True
        resp["landmarks"] = [{"x": round(l.x, 4), "y": round(l.y, 4), "z": round(l.z, 4)} for l in lms]

        if result.handedness:
            cat = result.handedness[0][0]
            resp["handedness"] = {"label": "Right" if cat.category_name == "Left" else "Left",
                                  "score": round(cat.score, 3)}

        raw = from_mediapipe(SimpleNamespace(landmark=lms), w, h)
        if raw is not None:
            bundle = models[active]
            if "keras" in bundle:      # 1D-CNN takes the raw (21, 3) landmarks
                probs = bundle["keras"](raw.reshape(1, 21, 3), training=False).numpy()[0]
            else:                      # sklearn models take the 93 engineered features
                probs = bundle["pipeline"].predict_proba(featurize(raw).reshape(1, -1))[0]
            idx = int(np.argmax(probs))
            resp["letter"] = bundle["classes"][idx]
            resp["confidence"] = round(float(probs[idx]), 4)

    resp["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonify(resp)


if __name__ == "__main__":
    print("=" * 60 + "\n  ASL Translator  ->  http://localhost:5000\n" + "=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
