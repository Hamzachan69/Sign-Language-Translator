"""
=============================================================================
  ASL Project - Full Environment Diagnostic
  Run this from your project root:
      python check_env.py
=============================================================================
"""

import sys
import subprocess

PASS  = "[PASS]"
FAIL  = "[FAIL]"
WARN  = "[WARN]"
INFO  = "[INFO]"
SEP   = "-" * 60

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ===========================================================================
# 1. Python version
# ===========================================================================
section("1. Python")
major, minor = sys.version_info[:2]
ver = f"{major}.{minor}"
if major == 3 and minor >= 9:
    print(f"  {PASS}  Python {ver}  ({sys.executable})")
else:
    print(f"  {FAIL}  Python {ver} is too old. Use Python 3.9+")

# ===========================================================================
# 2. Required packages
# ===========================================================================
section("2. Required Packages")

packages = {
    "tensorflow":     ("2.15.0", "2.15.0"),
    "keras":          ("2.15.0", "2.15.0"),
    "numpy":          ("1.26.0", None),
    "opencv-python":  ("4.0.0",  None),
    "scikit-learn":   ("1.0.0",  None),
    "matplotlib":     ("3.5.0",  None),
    "seaborn":        ("0.12.0", None),
    "pillow":         ("9.0.0",  None),
}

import importlib, importlib.metadata

all_ok = True
for pkg, (min_ver, exact_ver) in packages.items():
    # importlib name differs for some packages
    import_name = pkg.replace("-", "_").replace("opencv_python", "cv2")
    try:
        dist_name = pkg  # pip name
        installed = importlib.metadata.version(dist_name)
        from packaging.version import Version
        inst_v = Version(installed)
        min_v  = Version(min_ver)
        if inst_v < min_v:
            print(f"  {WARN}  {pkg:<20} installed={installed}  (need >= {min_ver})")
            all_ok = False
        else:
            print(f"  {PASS}  {pkg:<20} {installed}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {FAIL}  {pkg:<20} NOT INSTALLED")
        all_ok = False

if all_ok:
    print(f"\n  {PASS}  All packages look good.")
else:
    print(f"\n  {WARN}  Some packages need attention. Run the fix command below.")
    print(f"\n        pip install tensorflow==2.15.0 keras==2.15.0 numpy==1.26.4")
    print(f"             scikit-learn opencv-python matplotlib seaborn pillow")

# ===========================================================================
# 3. TensorFlow import + basic op
# ===========================================================================
section("3. TensorFlow Sanity Check")
try:
    import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    # run a tiny computation to confirm the graph engine works
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.reduce_sum(a).numpy()
    print(f"  {PASS}  tensorflow imported successfully  (v{tf.__version__})")
    print(f"  {PASS}  Basic tensor op: sum([1,2,3]) = {b:.1f}")
except Exception as e:
    print(f"  {FAIL}  TensorFlow import failed: {e}")

# ===========================================================================
# 4. GPU / CUDA detection & Active Device Placement
# ===========================================================================
section("4. GPU & CUDA Device Verification")

# 4a - nvidia-smi
hardware_gpu = None
smi_cmd = "nvidia-smi"
import shutil
if not shutil.which(smi_cmd):
    smi_cmd = r"C:\Windows\System32\nvidia-smi.exe"

try:
    result = subprocess.run(
        [smi_cmd, "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0 and result.stdout.strip():
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        hardware_gpu = parts[0]
        print(f"  {PASS}  Physical GPU detected   : {hardware_gpu}")
        print(f"  {INFO}  Driver version         : {parts[1]}")
        print(f"  {INFO}  Total VRAM             : {parts[2]}")
    else:
        print(f"  {WARN}  nvidia-smi not found or no GPU detected")
except FileNotFoundError:
    print(f"  {WARN}  nvidia-smi not in PATH - cannot check GPU driver")

# 4b - TF GPU detection & Active Test
try:
    build_info = tf.sysconfig.get_build_info()
    is_cuda_build = build_info.get("is_cuda_build", False)
    print(f"  {INFO}  TensorFlow CUDA build  : {is_cuda_build}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            print(f"  {PASS}  TensorFlow sees device : {g.name}")
        # Active execution test
        with tf.device("/device:GPU:0"):
            mat_test = tf.matmul(tf.ones([100, 100]), tf.ones([100, 100]))
            _ = mat_test.numpy()
        print(f"  {PASS}  Active tensor execution verified on GPU: {mat_test.device}")
        print(f"\n  {PASS}  GPU TRAINING IS ACTIVELY ENABLED AND FUNCTIONING!")
    else:
        print(f"\n  {FAIL}  TENSORFLOW IS NOT USING YOUR GPU (RUNNING ON CPU)")
        print(f"  {INFO}  Your physical GPU ({hardware_gpu or 'NVIDIA GPU'}) is ready,")
        print(f"         but TensorFlow on native Windows has no CUDA support in TF 2.11+.")
        print(f"  {INFO}  Run 'python check_gpu.py' for details on enabling RTX 3070 via WSL2.")
except Exception as e:
    print(f"  {FAIL}  GPU check failed: {e}")

# ===========================================================================
# 5. Dataset structure
# ===========================================================================
section("5. Dataset")
from pathlib import Path

dataset_root = Path("dataset") / \
    "ASL-HG American Sign Language Hand Gesture Image D" / \
    "ASL_HG_36000" / "asl_processed"

checks = {
    "asl_processed/train" : dataset_root / "train",
    "asl_processed/test"  : dataset_root / "test",
}

for label, path in checks.items():
    if path.exists():
        classes    = [d for d in path.iterdir() if d.is_dir()]
        img_count  = sum(len(list(c.glob("*.jpg")) + list(c.glob("*.png")))
                         for c in classes[:3]) * (len(classes) // 3 or 1)
        print(f"  {PASS}  {label:<30} {len(classes)} classes  (~{img_count:,} images)")
    else:
        print(f"  {FAIL}  {label:<30} NOT FOUND  (expected at: {path})")

# ===========================================================================
# 6. Model scripts
# ===========================================================================
section("6. Model Scripts")
scripts = [
    "models/utils/data_loader.py",
    "models/utils/metrics.py",
    "models/knn_classifier.py",
    "models/kmeans_classifier.py",
    "models/cnn_classifier.py",
    "models/run_all.py",
]
for s in scripts:
    p = Path(s)
    if p.exists():
        kb = p.stat().st_size / 1024
        print(f"  {PASS}  {s:<40}  ({kb:.1f} KB)")
    else:
        print(f"  {FAIL}  {s}  MISSING")

# ===========================================================================
# 7. Quick pipeline smoke-test
# ===========================================================================
section("7. Quick Pipeline Smoke-Test")
try:
    import numpy as np
    import sys; sys.path.insert(0, "models")
    from utils.data_loader import load_split, TEST_DIR
    from utils.metrics import print_metrics

    print(f"  {INFO}  Loading 10 test images per class for quick check ...")
    # Load only first 10 images per class by monkey-patching glob limit
    import cv2
    classes = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])
    images, labels = [], []
    for idx, cls in enumerate(classes[:5]):   # just 5 classes, 5 imgs each
        cls_dir = TEST_DIR / cls
        for fpath in list(cls_dir.glob("*.jpg"))[:5]:
            img = cv2.imread(str(fpath))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))
                images.append(img)
                labels.append(idx)
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels)
    print(f"  {PASS}  Loaded {len(X)} images  shape={X.shape}  dtype={X.dtype}")
except Exception as e:
    print(f"  {FAIL}  Pipeline smoke-test failed: {e}")

# ===========================================================================
# Final summary
# ===========================================================================
section("READY TO TRAIN")
print(f"""
  Run one of these commands from your project root:

    python models/run_all.py                 # all 3 models in sequence
    python models/run_all.py --model knn     # KNN only
    python models/run_all.py --model kmeans  # K-Means only
    python models/run_all.py --model cnn     # CNN only

  Results (metrics + confusion matrix PNGs) saved to:
    models/results/
""")

