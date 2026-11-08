"""
==============================================================================
  ASL Classifier - MediaPipe LANDMARKS + 1D Convolutional Neural Network
==============================================================================
The 21 landmarks are treated as a short sequence of 21 "steps" with 3 channels
(x, y, z). 1D convolutions slide over neighbouring joints, learning local finger
shapes (a bent knuckle, a spread pair of fingers) instead of looking at pixels.

HOW TO RUN
----------
  pip install tensorflow==2.15.0 mediapipe scikit-learn opencv-python joblib numpy matplotlib seaborn
  python models/cnn_classifier.py [--limit 200] [--rebuild]

GPU
---
  Used automatically if TensorFlow can see it (memory growth is enabled and the
  active device is printed). On native Windows, TF 2.11+ has no GPU support: use
  WSL2, otherwise it runs on CPU, which is still fine for a model this small.

OUTPUT
------
  models/results/cnn_landmark.keras
  models/results/cnn_classes.json
  models/results/cnn_confusion_matrix.png
  models/results/cnn_training_history.png

UNINSTALL: delete models/results/cnn_landmark.keras, cnn_classes.json, cnn_confusion_matrix.png, cnn_training_history.png
"""
import os, sys, time, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.model_selection import train_test_split
except ImportError as e:
    sys.exit(f"[FAIL] Missing package '{e.name}'.\n"
             f"       Fix: pip install tensorflow==2.15.0 mediapipe scikit-learn opencv-python joblib numpy matplotlib seaborn")

from utils.landmark_data import load_landmark_data, augment, RESULTS_DIR
from utils.metrics import print_metrics, plot_confusion_matrix, plot_training_history

MODEL_PATH = RESULTS_DIR / "cnn_landmark.keras"
CLASSES_PATH = RESULTS_DIR / "cnn_classes.json"
EPOCHS, BATCH, LR = 150, 256, 1e-3


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("\n  [INFO] No GPU visible to TensorFlow -> training on CPU (fine for this small model).")
        print("         Native Windows has no TF GPU support since 2.11; use WSL2 to enable it.")
        return
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"\n  [PASS] GPU active: {[g.name for g in gpus]}")


def build_model(n_classes):
    inp = keras.Input(shape=(21, 3), name="landmarks")
    x = layers.Conv1D(64, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)                       # keep joint positions (21 x 128)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inp, out, name="ASL_Landmark_CNN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 60 + "\n  Landmarks + 1D-CNN  -  ASL Classifier\n" + "=" * 60)
    configure_gpu()
    X_tr, y_tr, X_te, y_te, classes = load_landmark_data(args.limit, args.rebuild)

    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)
    Xa, ya = augment(Xa, ya, jitter=0.02)         # mirror + noisy copies
    Xa, Xv, Xt = (a.reshape(-1, 21, 3).astype("float32") for a in (Xa, Xv, X_te))

    model = build_model(len(classes))
    model.summary(line_length=70)
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cbs = [
        callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]
    t0 = time.time()
    hist = model.fit(Xa, ya, validation_data=(Xv, yv), epochs=EPOCHS, batch_size=BATCH,
                     callbacks=cbs, verbose=1)
    print(f"\n  Training complete in {(time.time()-t0)/60:.1f} min")
    CLASSES_PATH.write_text(json.dumps(classes))
    plot_training_history(hist.history, "Landmark 1D-CNN",
                          save_path=str(RESULTS_DIR / "cnn_training_history.png"))

    print("\n[4] Test-set evaluation ...\n")
    y_pred = np.argmax(model.predict(Xt, batch_size=512, verbose=0), axis=1)
    res = print_metrics(y_te, y_pred, classes, "Landmarks + 1D-CNN")
    plot_confusion_matrix(res["confusion_matrix"], classes, "Landmarks + 1D-CNN",
                          save_path=str(RESULTS_DIR / "cnn_confusion_matrix.png"))
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
