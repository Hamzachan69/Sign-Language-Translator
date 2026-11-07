"""
=============================================================================
  ASL Hand-Gesture Classifier - Convolutional Neural Network (CNN)
  GPU-accelerated: optimised for NVIDIA RTX 3070 (8 GB VRAM)
=============================================================================

HOW TO RUN
----------
  python models/cnn_classifier.py

WHAT IT DOES
------------
  1. Configures the RTX 3070 with memory-growth + mixed precision (FP16)
     for maximum GPU throughput.
  2. Loads train / test images as (64, 64, 3) tensors via tf.data pipelines.
  3. Builds a CNN with Keras preprocessing layers for data augmentation
     (applied only during training, not at inference time):
         RandomFlip / RandomRotation / RandomZoom / RandomContrast
       + Conv(32)  -> BN -> Conv(64)  -> BN -> MaxPool -> Dropout
       + Conv(128) -> BN -> Conv(128) -> BN -> MaxPool -> Dropout
       + Conv(256) -> BN -> GlobalAvgPool
       + Dense(256) -> Dropout -> Dense(36, softmax)
  4. Trains with: Adam, label-smoothing cross-entropy,
                  LR scheduler, early stopping.
  5. Evaluates on the test set and prints:
       Accuracy / Precision / Recall / F1-Score / Confusion Matrix
  6. Saves the trained model weights + both plots as PNGs.

OUTPUT FILES
------------
  models/results/cnn_best_model.keras
  models/results/cnn_training_history.png
  models/results/cnn_confusion_matrix.png
"""

import os, sys, time
import numpy as np
from pathlib import Path

# ── make sibling imports work when script is run directly ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Suppress verbose TF startup logs (keep warnings/errors only)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from utils.data_loader import TRAIN_DIR, TEST_DIR, load_split, IMG_SIZE
from utils.metrics import print_metrics, plot_confusion_matrix, plot_training_history


# ─── GPU setup ────────────────────────────────────────────────────────────────
def configure_gpu():
    """
    Configure RTX 3070 for maximum performance:
      - Memory growth  : avoid allocating all 8 GB VRAM upfront
      - Mixed precision: float16 compute / float32 variables
                         -> ~2x throughput on Tensor Cores
      - Explicit device verification
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Mixed precision - float16 on RTX Tensor Cores
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        
        # Test active computation on GPU
        with tf.device("/device:GPU:0"):
            t = tf.matmul(tf.ones([64, 64]), tf.ones([64, 64]))
            _ = t.numpy()

        print(f"\n  [PASS] GPU ACCELERATION ACTIVE")
        print(f"         Device count : {len(gpus)}")
        for i, g in enumerate(gpus):
            print(f"         Device {i}     : {g.name}")
        print( "         Precision    : mixed_float16 (Tensor Core acceleration active)")
        print(f"         Test tensor  : verified on {t.device}")
        return True
    else:
        print("\n  [INFO] No GPU device found by TensorFlow on this system.")
        print("         TensorFlow is executing on: CPU")
        print("         (Tip: Run 'python check_gpu.py' in terminal for details on RTX 3070 setup)")
        return False


# ─── hyper-parameters ─────────────────────────────────────────────────────────
EPOCHS        = 80
BATCH_SIZE_GPU = 128   # large batch saturates RTX 3070 Tensor Cores
BATCH_SIZE_CPU = 64
LR_INITIAL    = 1e-3
LR_MIN        = 1e-6
DROPOUT_RATE  = 0.4
PATIENCE      = 8
RESULTS_DIR   = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_PATH    = str(RESULTS_DIR / "cnn_best_model.keras")


# ─── tf.data pipeline ─────────────────────────────────────────────────────────
# IMPORTANT: Augmentation that uses Keras layers (RandomRotation, RandomZoom,
# etc.) must NOT be called inside ds.map() because tf.function traces the
# function and those layers create tf.Variables on each call -> ValueError.
#
# Solution: use ONLY native tf.image ops in ds.map(), and place Keras
# preprocessing layers at the top of the MODEL GRAPH where Keras manages
# their variables correctly. Those layers auto-disable at inference time.

def _pixel_augment(image, label):
    """Pixel-level augmentation using native tf.image ops (safe in tf.function)."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.85, 1.15)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def make_tf_dataset(X, y, batch_size, shuffle=False, augment_data=False):
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32"), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42,
                        reshuffle_each_iteration=True)
    if augment_data:
        ds = ds.map(_pixel_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─── model architecture ───────────────────────────────────────────────────────
def build_cnn(input_shape, n_classes, dropout_rate):
    """
    Layout
    ------
    [Augmentation head - Keras preprocessing, training-only]
      RandomFlip -> RandomRotation(8%) -> RandomZoom(10%) -> RandomContrast(15%)

    Block 1  : Conv2D(32)  -> BN -> ReLU
               Conv2D(64)  -> BN -> ReLU -> MaxPool(2) -> Dropout
    Block 2  : Conv2D(128) -> BN -> ReLU
               Conv2D(128) -> BN -> ReLU -> MaxPool(2) -> Dropout
    Block 3  : Conv2D(256) -> BN -> ReLU -> GlobalAvgPool
    Head     : Dense(256)  -> BN -> ReLU -> Dropout
               Dense(36)   -> Softmax  [float32 for numerical stability]

    Mixed-precision note
    --------------------
    When mixed_float16 is active, all Conv/Dense compute in float16 but
    variables stay float32. The final Dense + softmax is explicitly cast
    back to float32 to avoid NaN in the loss.
    """
    inp = keras.Input(shape=input_shape, name="image", dtype="float32")

    # ── Augmentation (Keras layers - managed by the model, not by ds.map) ──
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomRotation(0.08)(x)      # +/- ~29 deg
    x = layers.RandomZoom(0.10)(x)          # +/- 10 %
    x = layers.RandomContrast(0.15)(x)      # random contrast shift

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── Block 3 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    # ── Head ─────────────────────────────────────────────────────────────────
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Cast to float32 before softmax - required for FP16 mixed precision
    x = layers.Activation("linear", dtype="float32")(x)
    out = layers.Dense(n_classes, activation="softmax",
                       dtype="float32", name="predictions")(x)

    return keras.Model(inputs=inp, outputs=out, name="ASL_CNN")


def main():
    print("\n" + "=" * 60)
    print("  Convolutional Neural Network  -  ASL Classifier")
    print("=" * 60)

    # ── Configure RTX 3070 ────────────────────────────────────────────────────
    on_gpu     = configure_gpu()
    batch_size = BATCH_SIZE_GPU if on_gpu else BATCH_SIZE_CPU

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4]  Loading dataset ...")
    X_train, y_train, classes = load_split(TRAIN_DIR, flatten=False)
    X_test,  y_test,  _       = load_split(TEST_DIR,  flatten=False)
    n_classes = len(classes)
    print(f"       Train : {X_train.shape}   Test : {X_test.shape}")
    print(f"       Classes ({n_classes}): {', '.join(classes)}")
    print(f"       Batch size : {batch_size}")

    # Validation split (10 % of training data)
    rng      = np.random.default_rng(42)
    idx      = rng.permutation(len(X_train))
    val_size = int(0.1 * len(X_train))
    val_idx, tr_idx = idx[:val_size], idx[val_size:]

    X_tr,  y_tr  = X_train[tr_idx],  y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    train_ds = make_tf_dataset(X_tr,   y_tr,   batch_size, shuffle=True,  augment_data=True)
    val_ds   = make_tf_dataset(X_val,  y_val,  batch_size, shuffle=False, augment_data=False)
    test_ds  = make_tf_dataset(X_test, y_test, batch_size, shuffle=False, augment_data=False)

    # ── 2. Build & compile ────────────────────────────────────────────────────
    print("\n[2/4]  Building CNN ...")
    model = build_cnn(input_shape, n_classes, DROPOUT_RATE)
    model.summary(line_length=70)

    try:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, label_smoothing=0.1
        )
    except TypeError:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_INITIAL),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # ── 3. Callbacks ──────────────────────────────────────────────────────────
    cb_list = [
        callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=LR_MIN, verbose=1,
        ),
    ]

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print(f"\n[3/4]  Training for up to {EPOCHS} epochs ...\n")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb_list,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed / 60:.1f} min")

    hist_path = str(RESULTS_DIR / "cnn_training_history.png")
    plot_training_history(history.history, "CNN", save_path=hist_path)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/4]  Evaluating on test set ...\n")
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    results = print_metrics(y_test, y_pred, classes, "CNN")

    cm_path = str(RESULTS_DIR / "cnn_confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], classes,
                          "CNN", save_path=cm_path)

    print(f"\n  Done.  Results saved to: {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
