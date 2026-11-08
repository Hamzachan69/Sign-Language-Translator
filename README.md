# 🖐️ American Sign Language (ASL) Real-Time Recognition & Translation System

An end-to-end, multi-model Computer Vision and Machine Learning system for real-time American Sign Language (ASL) fingerspelling recognition, word assembly, and live speech translation.

The system is built on **Google MediaPipe Tasks API** for robust 3D hand skeletal landmark tracking, a **93-dimensional geometric feature engineering pipeline**, and a **multi-model inference engine** supporting five distinct algorithms (**SVM**, **MLP**, **KNN**, **K-Means**, and **1D-CNN**) across **36 ASL classes** (digits `0–9` and alphabet `A–Z`).

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [The Landmark Revolution (Why Landmarks over Pixels?)](#-the-landmark-revolution-why-landmarks-over-pixels)
- [How The Program Works (End-to-End Pipeline)](#-how-the-program-works-end-to-end-pipeline)
  - [1. Client-Side Video Capture & Offscreen Compression](#1-client-side-video-capture--offscreen-compression)
  - [2. MediaPipe Hand Landmark Detection (Tasks API)](#2-mediapipe-hand-landmark-detection-tasks-api)
  - [3. Normalization & 93-Dimensional Feature Engineering](#3-normalization--93-dimensional-feature-engineering)
  - [4. Multi-Model Inference Dispatch](#4-multi-model-inference-dispatch)
  - [5. Interactive Frontend, Hold-to-Confirm & TTS](#5-interactive-frontend-hold-to-confirm--tts)
- [How Each Algorithm Works (Theory & Training)](#-how-each-algorithm-works-theory--training)
  - [1. Support Vector Machine (SVM) — Default Production Engine](#1-support-vector-machine-svm--default-production-engine)
  - [2. Multi-Layer Perceptron (MLP) — Deep Feed-Forward Network](#2-multi-layer-perceptron-mlp--deep-feed-forward-network)
  - [3. K-Nearest Neighbors (KNN) — Instance-Based Metric Learning](#3-k-nearest-neighbors-knn--instance-based-metric-learning)
  - [4. K-Means Clustering — Unsupervised Learning + Majority Voting](#4-k-means-clustering--unsupervised-learning--majority-voting)
  - [5. 1D Convolutional Neural Network (1D-CNN) — Topological Sequence Model](#5-1d-convolutional-neural-network-1d-cnn--topological-sequence-model)
- [Benchmark & Evaluation Metrics](#-benchmark--evaluation-metrics)
- [Project Directory Structure](#-project-directory-structure)
- [Tech Stack & Dependencies](#-tech-stack--dependencies)
- [Setup & Getting Started](#-setup--getting-started)
  - [Installation](#installation)
  - [Environment Verification](#environment-verification)
- [Training the Models](#-training-the-models)
- [Running the Live Application](#-running-the-live-application)
- [Standalone Diagnostic & Benchmark Utilities](#-standalone-diagnostic--benchmark-utilities)
- [Files Cleanup & Maintenance Guide](#-files-cleanup--maintenance-guide)

---

## 🏛️ Overview & Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 BROWSER CLIENT (UI)                    │
                  │  • Webcam stream via getUserMedia (1280x720)           │
                  │  • Offscreen canvas compression -> JPEG base64 payload │
                  │  • Overlay Canvas: 21 skeletal joints & connecting bones│
                  │  • Hold-to-confirm duration slider (0.2s - 5.0s)       │
                  │  • Dynamic model switcher (queried from /api/models)   │
                  │  • Sentence builder & Web Speech API TTS voice readout │
                  └───────────────────────────┬────────────────────────────┘
                                              │ HTTP POST /api/predict
                                              │ (frame payload + model id)
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                 FLASK SERVER (PYTHON)                  │
                  │  1. Decode base64 buffer -> OpenCV BGR Frame           │
                  │  2. MediaPipe HandLandmarker (Tasks API asset)         │
                  │  3. Selfie-mode handedness correction (Left ↔ Right)   │
                  │  4. Normalize landmarks: wrist = origin (0,0,0), scale │
                  │  5. Engineered features (93-D) or raw joint sequence   │
                  └───────────────────────────┬────────────────────────────┘
                                              │
        ┌───────────────────┬─────────────────┼─────────────────┬───────────────────┐
        ▼                   ▼                 ▼                 ▼                   ▼
┌──────────────┐    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    ┌──────────────┐
│  SVM (RBF)   │    │  MLP (Dense) │   │  KNN (k=5)   │   │K-Means(k=108)│    │   1D-CNN     │
│ StandardScaler│   │ StandardScaler│  │StandardScaler│   │StandardScaler│    │ Conv1D x 3   │
│   + SVC(RBF) │    │+ MLP(256,128)│   │  + Dist-KNN  │   │+ KMeans-Vote │    │+ BatchNorm   │
│Acc: 100.00%  │    │ Acc: 99.97%  │   │ Acc: 99.97%  │   │ Acc: 79.65%  │    │ Acc: ~99.0%  │
└───────┬──────┘    └───────┬──────┘   └───────┬──────┘   └───────┬──────┘    └───────┬──────┘
        └───────────────────┴─────────────────┼─────────────────┴───────────────────┘
                                              │ Response JSON
                                              │ {letter, confidence, landmarks,
                                              │  handedness, latency_ms}
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Updates active letter display, confidence meter,       │
                  │ progress ring, latency readout, and canvas skeleton    │
                  └────────────────────────────────────────────────────────┘
```

---

## 💡 The Landmark Revolution (Why Landmarks over Pixels?)

Traditional sign language models classify raw cropped camera pixels. This approach introduces severe real-world failure modes:
1. **Lighting & Skin-Tone Sensitivity:** Ambient room shadows, dim lamps, and skin pigment variations shift raw RGB distributions drastically.
2. **Background Clutter:** Objects, furniture, and clothing behind the hand distort convolutional activations.
3. **High Dimensionality:** A $64 \times 64 \times 3$ crop requires processing 12,288 dimensions per frame.
4. **Sign Ambiguity:** Letters with similar silhouettes (e.g., `M`, `N`, `S`, `T` or `U`, `V`, `R`) produce overlapping pixel profiles.

### The Modern Solution:
The system shifts the entire architecture to **Google MediaPipe Hand Landmarks** and **kinematic feature engineering**:
- The hand is converted into **21 discrete 3D spatial joints** $(x, y, z)$.
- Pixels are discarded; only anatomical relationships (joint hinge angles, finger curl amounts, fingertip-to-fingertip contact distances) are evaluated.
- **Left/Right Hand Agnostic:** Applying horizontal reflection ($x \to -x$) during training makes all classifiers seamlessly recognize gestures from both left and right hands.
- **Blazing Performance:** Feature vectors shrink from 12,288 down to **93 numbers**, reducing CPU inference latency to **under 1 ms**.

---

## 🔄 How The Program Works (End-to-End Pipeline)

### 1. Client-Side Video Capture & Offscreen Compression
- The user launches the translator at `http://localhost:5000/translate`.
- Camera feed is acquired via `navigator.mediaDevices.getUserMedia` at 1280×720 resolution.
- An offscreen canvas extracts frames at 320×240 JPEG compression (`quality=0.75`) and transmits them via `POST /api/predict`.

### 2. MediaPipe Hand Landmark Detection (Tasks API)
- The Flask backend receives the payload and decodes it via `cv2.imdecode`.
- MediaPipe's **Tasks API** (`vision.HandLandmarker` utilizing `models/hand_landmarker.task`) locates the 21 3D hand keypoints (wrist, MCP, PIP, DIP, and fingertip coordinates).
- **Selfie Mode Inversion:** Raw webcam video is mirrored on-screen for user comfort. The server inverts the MediaPipe handedness classification (`Left` becomes `Right` and vice versa) so the user's physical hand is accurately recognized.

### 3. Normalization & 93-Dimensional Feature Engineering
Hand landmarks are transformed via [`models/utils/landmark_features.py`](file:///c:/GitHub/Sign%20Language%20Translator/models/utils/landmark_features.py):
1. **Wrist-Origin Translation:** All 21 joint points are translated so the wrist joint (index 0) is situated at $(0, 0, 0)$, providing **100% position invariance**.
2. **Scale Normalization:** Points are divided by the maximum Euclidean distance from the wrist to any knuckle/tip, ensuring **100% camera-distance invariance**.
3. **Geometric Features (93 total):**
   - **63 Raw Coordinates:** 21 normalized $(x, y, z)$ coordinates.
   - **10 Fingertip Distances:** Pairwise Euclidean distances between all 5 fingertips ($\binom{5}{2} = 10$).
   - **15 Joint Bend Angles:** Cosine angles across the 3 anatomical hinge joints per finger ($5 \times 3 = 15$).
   - **5 Finger Curl Distances:** Direct distance from each fingertip to its own MCP knuckle, measuring individual finger curling.

### 4. Multi-Model Inference Dispatch
The 93-dimensional feature vector (or raw $21 \times 3$ sequence for 1D-CNN) is routed to the selected model:
- **SVM:** Evaluates probability via RBF kernel boundary margins.
- **MLP:** Feeds through a 2-hidden-layer neural network with Softmax output.
- **KNN:** Computes distance-weighted votes across the $K=5$ closest neighbors.
- **K-Means:** Finds the closest of 108 centroids, maps to a class via majority vote, and calculates soft distance confidence.
- **1D-CNN:** Slides 1D convolutions across joint chains to capture spatial joint dependencies.

### 5. Interactive Frontend, Hold-to-Confirm & TTS
- **Canvas Skeleton:** Overlays the 21 landmark joints with color-coded fingers and connecting bones on top of the live video.
- **Hold-to-Confirm Duration Slider (0.2s – 5.0s, default 3.0s):** Requires the user to hold a sign steadily before appending the letter to the sentence, filtering out transitional hand movements.
- **Text-to-Speech (TTS):** Uses the browser's native Web Speech API (`window.speechSynthesis`) to speak out recognized words or completed sentences with one click.
- **Dynamic Model Selector:** The dropdown dynamically reflects which models are currently trained and saved on disk via `/api/models`.

---

## 🧠 How Each Algorithm Works (Theory & Training)

### 1. Support Vector Machine (SVM) — Default Production Engine
*File: [models/svm_classifier.py](file:///c:/GitHub/Sign%20Language%20Translator/models/svm_classifier.py)*

The SVM is the primary production engine, achieving flawless separation on the 93 geometric features.
- **Kernel Function:** Radial Basis Function (RBF):
  $$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$
- **Optimization:** Evaluates soft margin penalties $C \in \{3, 10, 30\}$ on a validation split, selecting $C=10$ with `gamma='scale'`.
- **Probabilities:** Fitted with `probability=True` (Platt scaling) to output calibrated prediction confidences.
- **Performance:** **100.00% Test Accuracy**, 0.54 ms latency.

---

### 2. Multi-Layer Perceptron (MLP) — Deep Feed-Forward Network
*File: [models/mlp_classifier.py](file:///c:/GitHub/Sign%20Language%20Translator/models/mlp_classifier.py)*

A lightweight, fully connected deep neural network operating on normalized features:
- **Architecture:**
  $$\mathbf{x} \in \mathbb{R}^{93} \xrightarrow{\text{Dense}(256) + \text{ReLU}} \mathbf{h}_1 \xrightarrow{\text{Dense}(128) + \text{ReLU}} \mathbf{h}_2 \xrightarrow{\text{Dense}(36) + \text{Softmax}} \hat{\mathbf{y}}$$
- **Regularization & Stopping:** Trained with Adam, an L2 penalty ($\alpha = 10^{-3}$), and automated early stopping (`patience=15`) monitoring validation loss.
- **Performance:** **99.97% Test Accuracy**, ultra-fast **0.003 ms latency** on CPU.

---

### 3. K-Nearest Neighbors (KNN) — Instance-Based Metric Learning
*File: [models/knn_classifier.py](file:///c:/GitHub/Sign%20Language%20Translator/models/knn_classifier.py)*

A non-parametric classifier operating in standardized 93-dimensional Euclidean space:
- **Distance Weighting:** Inverse-distance weighting ($w_i = 1 / d_i$) ensures closer training gestures dominate over distant neighbors.
- **Hyperparameter Tuning:** Grid searches $K \in \{3, 5, 7, 9, 15\}$; $K=5$ yields the highest validation score.
- **Performance:** **99.97% Test Accuracy**, 0.44 ms latency.

---

### 4. K-Means Clustering — Unsupervised Learning + Majority Voting
*File: [models/kmeans_classifier.py](file:///c:/GitHub/Sign%20Language%20Translator/models/kmeans_classifier.py)*

Evaluates how well hand gestures partition without human labels:
- **Overclustering ($K=108$):** Rather than forcing 36 clusters for 36 letters, K-Means uses 108 centroids (averaging 3 clusters per letter). This allows variations in wrist angle or individual signing styles to cluster naturally.
- **Majority-Vote Labeling:** After unsupervised clustering completes on the training set, each cluster is assigned the most frequent ground-truth class among its member samples.
- **Soft Confidence Estimation:**
  $$P(C_j) = \frac{\exp\left(-d_j + \min(\mathbf{d})\right)}{\sum_m \exp\left(-d_m + \min(\mathbf{d})\right)}$$
- **Performance:** **79.65% Test Accuracy**, 0.007 ms latency.

---

### 5. 1D Convolutional Neural Network (1D-CNN) — Topological Sequence Model
*File: [models/cnn_classifier.py](file:///c:/GitHub/Sign%20Language%20Translator/models/cnn_classifier.py)*

Treats the 21 landmarks as an anatomical sequence of 21 steps with 3 spatial channels $(x, y, z)$:
- **Architecture:** 3 sequential 1D Convolution blocks (`Conv1D(64)`, `Conv1D(128)`, `Conv1D(128)`) with Batch Normalization slide over consecutive joint chains, learning localized knuckle bend patterns.
- **Classification Head:** Flatten $\to$ Dropout(0.4) $\to$ Dense(256) $\to$ Softmax(36).
- **Performance:** **~99.0% Test Accuracy**, ~1.2 ms latency.

---

## 📊 Benchmark & Evaluation Metrics

Evaluated on the held-out test split (6,982 test samples across all 36 classes) using [`show_accuracy.py`](file:///c:/GitHub/Sign%20Language%20Translator/show_accuracy.py):

| Model Architecture | Model Paradigm | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Landmarks + SVM** | RBF Kernel ($C=10, \gamma=\text{'scale'}$) | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **0.54 ms / sample** |
| **Landmarks + MLP** | Neural Net $(256 \to 128 \to 36)$ | **99.97%** | **99.97%** | **99.97%** | **99.97%** | **0.003 ms / sample** |
| **Landmarks + KNN** | Instance-Based ($K=5$, Distance-weighted) | **99.97%** | **99.97%** | **99.97%** | **99.97%** | **0.44 ms / sample** |
| **Landmarks + 1D-CNN**| Deep Learning (Sequence Conv1D) | **~99.00%** | **~99.10%** | **~99.00%** | **~99.00%** | **1.20 ms / sample** |
| **Landmarks + K-Means**| Unsupervised ($K=108$ + Majority Vote) | **79.65%** | **79.16%** | **79.65%** | **78.92%** | **0.007 ms / sample** |

---

## 📁 Project Directory Structure

```
Sign Language Translator/
│
├── app.py                         # Production Flask server & real-time prediction API
├── show_accuracy.py               # Benchmark evaluation report generator
├── HandTracker.py                 # Standalone OpenCV desktop hand skeleton visualizer
├── check_env.py                   # Environment and package verification script
├── check_gpu.py                   # CUDA / GPU hardware diagnostic script
├── prompt.md                      # Slide-by-slide technical presentation briefing
├── README.md                      # Full project documentation & architecture reference
│
├── dataset/                       # ASL dataset directory
│   └── asl_processed/
│       ├── train/                 # 36 class folders (0-9, A-Z) with training images
│       └── test/                  # 36 class folders with held-out test images
│
├── models/
│   ├── svm_classifier.py          # SVM (RBF) model training script
│   ├── mlp_classifier.py          # MLP neural network training script
│   ├── knn_classifier.py          # KNN (distance-weighted) training script
│   ├── kmeans_classifier.py       # MiniBatchKMeans (K=108) training script
│   ├── cnn_classifier.py          # 1D-CNN joint sequence training script
│   ├── run_all.py                 # Unified sequential runner for all classifiers
│   ├── hand_landmarker.task       # MediaPipe Tasks API pretrained binary asset
│   │
│   ├── results/                   # Serialized model artifacts & evaluation plots
│   │   ├── svm_model.joblib
│   │   ├── svm_confusion_matrix.png
│   │   ├── mlp_model.joblib
│   │   ├── mlp_confusion_matrix.png
│   │   ├── knn_model.joblib
│   │   ├── knn_confusion_matrix.png
│   │   ├── kmeans_model.joblib
│   │   ├── kmeans_confusion_matrix.png
│   │   ├── cnn_landmark.keras
│   │   ├── cnn_classes.json
│   │   ├── landmark_features_train.npz # Cached training landmark vectors
│   │   └── landmark_features_test.npz  # Cached test landmark vectors
│   │
│   └── utils/
│       ├── landmark_features.py   # Core contract: from_mediapipe, mirror, featurize
│       ├── landmark_data.py       # Landmark extractor & NPZ cache manager
│       ├── landmark_models.py     # KMeansVoteClassifier custom estimator
│       ├── metrics.py             # Evaluation printing & confusion matrix plotter
│       └── data_loader.py         # Standardized image loader utility
│
├── templates/
│   ├── welcome.html               # Welcome landing page
│   └── translator.html            # Main real-time translation web interface
│
└── static/                        # Static styling and frontend scripts
```

---

## 🛠️ Tech Stack & Dependencies

- **Computer Vision & Tracking:** Google MediaPipe Tasks API (`vision.HandLandmarker`), OpenCV (`cv2`)
- **Machine Learning & Modeling:** Scikit-Learn (`SVC`, `MLPClassifier`, `KNeighborsClassifier`, `MiniBatchKMeans`, `StandardScaler`, `Pipeline`)
- **Deep Learning:** TensorFlow 2.15 / Keras (1D-CNN kinematic sequence model)
- **Backend Server:** Flask, Joblib, NumPy
- **Frontend UI:** HTML5, CSS3, JavaScript (ES6+), Tailwind CSS, Canvas API
- **Audio & Accessibility:** Native Web Speech API (`window.speechSynthesis`)
- **Plotting & Visualization:** Matplotlib, Seaborn

---

## 🚀 Setup & Getting Started

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/Sign-Language-Translator.git
   cd "Sign Language Translator"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install required packages:**
   ```bash
   pip install flask mediapipe opencv-python scikit-learn joblib numpy matplotlib seaborn
   ```

### Environment Verification
Run the built-in diagnostic utility:
```bash
python check_env.py
```

---

## 🏋️ Training the Models

You can train any model individually, or run the complete suite:

```bash
# Option 1: Train the SVM (Default production model)
python models/svm_classifier.py

# Option 2: Train the MLP neural network
python models/mlp_classifier.py

# Option 3: Train the KNN classifier
python models/knn_classifier.py

# Option 4: Train the K-Means clustering pipeline
python models/kmeans_classifier.py

# Option 5: Train all models sequentially
python models/run_all.py --model all
```

> [!TIP]
> Landmark feature extraction is performed once and cached in `models/results/landmark_features_{train,test}.npz`. Subsequent trainings load instantly from cache in seconds!

---

## 💻 Running the Live Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```
   *The server auto-downloads `models/hand_landmarker.task` on first boot if missing.*

2. **Open the translator in your browser:**
   Navigate to:
   ```
   http://localhost:5000
   ```

3. **Using the System:**
   - Click **"Start Now"** to enter the translation suite.
   - Allow camera access when prompted.
   - Select your preferred model from the dropdown (SVM, MLP, KNN, K-Means, or 1D-CNN).
   - Form signs in view of your camera — the 21-joint skeleton will appear with color-coded fingers.
   - Adjust the **Hold Timer Slider** (0.2s – 5.0s) to calibrate sign commitment speed.
   - Click **"Speak"** or press Space to voice the accumulated sentence.

---

## 🔬 Standalone Diagnostic & Benchmark Utilities

### 1. Benchmark Evaluation Report (`show_accuracy.py`)
Run the unified benchmark report from your terminal:
```bash
python show_accuracy.py
```

### 2. Standalone Hand Skeleton Visualizer (`HandTracker.py`)
Launch a high-framerate desktop window visualizing 21 landmarks, finger colors, and joint index IDs:
```bash
python HandTracker.py
# Press 'Q' to quit
```

---

## 🧹 Files Cleanup & Maintenance Guide

To keep the repository clean and avoid shipping redundant files, the following legacy artifacts can be safely deleted:
- `models/results/cnn_best_model.keras`: Legacy 2D pixel-based CNN (~12.4 MB), superseded by `cnn_landmark.keras`.
- `models/results/landmark_model.joblib`: Legacy single-file test artifact, superseded by the individual classifier models.
- `UI/stitch_asl_sign_language_translator.zip`: Redundant zip archive of the `UI/` folder.
- `venv_wsl/`: Local WSL virtual environment directory (if present).
