# 🖐️ American Sign Language (ASL) Real-Time Recognition & Translation System

An end-to-end, multi-model Computer Vision and Deep Learning system for real-time American Sign Language (ASL) fingerspelling recognition and live text translation.

The project features a **full-stack web application** (Flask + Tailwind CSS + HTML5 Canvas), **real-time hand skeletal tracking** via Google MediaPipe, and a **multi-model inference engine** supporting deep learning (CNN) alongside classic machine learning baselines (KNN, K-Means) across **36 ASL classes** (digits `0–9` and alphabet `A–Z`).

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [How The Program Works (End-to-End Pipeline)](#-how-the-program-works-end-to-end-pipeline)
  - [1. Client-Side Video Capture & Canvas Rendering](#1-client-side-video-capture--canvas-rendering)
  - [2. Google MediaPipe Hand Landmark Detection](#2-google-mediapipe-hand-landmark-detection)
  - [3. Dynamic Region of Interest (ROI) Extraction](#3-dynamic-region-of-interest-roi-extraction)
  - [4. Multi-Model Inference & Dispatch](#4-multi-model-inference--dispatch)
  - [5. Interactive Frontend Features](#5-interactive-frontend-features)
- [How Each Algorithm Works (Theory & Training)](#-how-each-algorithm-works-theory--training)
  - [1. Convolutional Neural Network (CNN) - Deep Learning](#1-convolutional-neural-network-cnn---deep-learning)
  - [2. K-Nearest Neighbors (KNN) - Instance-Based Learning](#2-k-nearest-neighbors-knn---instance-based-learning)
  - [3. K-Means Clustering - Unsupervised Learning](#3-k-means-clustering---unsupervised-learning)
- [Comparative Performance Benchmark](#-comparative-performance-benchmark)
- [Tech Stack & Tools Used So Far](#-tech-stack--tools-used-so-far)
- [Project Directory Structure](#-project-directory-structure)
- [Setup & Getting Started](#-setup--getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Verification](#environment-verification)
- [Training the Models](#-training-the-models)
- [Running the Application](#-running-the-application)
- [Standalone Diagnostic & Utility Tools](#-standalone-diagnostic--utility-tools)

---

## 🏛️ Overview & Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 BROWSER CLIENT (UI)                    │
                  │  • Webcam stream via getUserMedia                      │
                  │  • Overlay Canvas: 21 skeletal joints & connections    │
                  │  • Stream loop: 12-15 FPS base64 JPEG payload          │
                  │  • TTS Speech Synthesis & Sentence Construction        │
                  └───────────────────────────┬────────────────────────────┘
                                              │ HTTP POST /api/predict
                                              │ (base64 frame + selected model)
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                 FLASK SERVER (PYTHON)                  │
                  │  1. Decode base64 buffer -> OpenCV BGR Frame           │
                  │  2. MediaPipe Hands: Extract 21 3D landmarks           │
                  │  3. ROI Extraction: Square clamp, pad, resize (64x64)  │
                  │  4. Color conversion: BGR -> RGB, normalize [0, 1]     │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
            ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
            │   CNN MODEL      │    │    KNN MODEL     │    │   K-MEANS MODEL  │
            │  (Deep Learning) │    │(StandardScaler + │    │(StandardScaler + │
            │  TensorFlow/Keras│    │   PCA(150) +     │    │   PCA(150) +     │
            │  Input: (64,64,3)│    │   KNN Classifier)│    │ MiniBatchKMeans) │
            │  Accuracy: ~99%  │    │  Accuracy: 81.4% │    │  Accuracy: 24.6% │
            └─────────┬────────┘    └─────────┬────────┘    └─────────┬────────┘
                      └───────────────────────┼───────────────────────┘
                                              │ Response JSON
                                              │ {letter, confidence, landmarks,
                                              │  handedness, latency_ms}
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Updates predictions, confidence bar, audio readout,   │
                  │ hand tracking overlay, and live latency metrics        │
                  └────────────────────────────────────────────────────────┘
```

---

## 🔄 How The Program Works (End-to-End Pipeline)

### 1. Client-Side Video Capture & Canvas Rendering
- The user accesses the web app at `http://localhost:5000/translate`.
- The browser requests webcam access using `navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })`.
- A continuous rendering loop grabs video frames, writes them to a hidden offscreen `<canvas>`, compresses the frame into a JPEG data URL, and posts it asynchronously to `/api/predict`.

### 2. Google MediaPipe Hand Landmark Detection
- The backend decodes the incoming JPEG into a NumPy array with OpenCV (`cv2.imdecode`).
- Frame is converted from BGR to RGB and passed to **MediaPipe Hands** (`mp.solutions.hands.Hands`).
- MediaPipe locates up to 21 distinct 3D landmarks on the hand (wrist, palm, thumb MCP/IP/tip, index MCP/PIP/DIP/tip, etc.).
- MediaPipe also evaluates **handedness** (`Left` or `Right`). Because webcam streams are mirrored for user comfort (selfie mode), the server adjusts the handedness tag to reflect the true hand position.

### 3. Dynamic Region of Interest (ROI) Extraction
To ensure high classification accuracy regardless of where the hand is situated in the camera frame:
- The system computes the minimum and maximum coordinates of all 21 detected landmarks `(x_min, x_max, y_min, y_max)`.
- It calculates the bounding box center `(cx, cy)` and computes a square bounding box by taking `side = max(width, height) + 2 * padding` (where padding = 25px).
- **Aspect Ratio Preservation & Border Padding**: If the hand is close to the frame boundary, clipping can cause the crop to become rectangular. The pipeline uses `cv2.copyMakeBorder` with black pixels (`cv2.BORDER_CONSTANT`) so the extracted patch remains an undistorted 1:1 square.
- The cropped hand image is converted to **RGB**, resized to **64 × 64 pixels** using `cv2.INTER_AREA`, and normalized to floating point values in `[0.0, 1.0]`.

### 4. Multi-Model Inference & Dispatch
The normalized 64×64 patch is handed off to the user-selected model in the model registry:
- **CNN**: Accepts `(1, 64, 64, 3)` tensor $\rightarrow$ outputs softmax probabilities across all 36 classes $\rightarrow$ takes `argmax` and max probability as confidence.
- **KNN**: Flattens patch to `(1, 12288)` vector $\rightarrow$ passes through `StandardScaler` $\rightarrow$ projects onto 150 components with `PCA` $\rightarrow$ computes class probabilities with `predict_proba`.
- **K-Means**: Flattens patch to `(1, 12288)` vector $\rightarrow$ standardizes and applies PCA(150) $\rightarrow$ predicts nearest cluster ID $\rightarrow$ maps cluster ID to true class via majority-vote mapping table (`pipe.cluster_map_`) $\rightarrow$ generates soft confidence from centroid distance.

### 5. Interactive Frontend Features
- **Live Skeletal Visualization**: The returned 21 landmarks are drawn onto the live camera preview with color-coded fingers (orange for thumb, green for index, rose for middle, purple for ring, cyan for pinky) and connecting bones.
- **Hold-to-Confirm / Sentence Builder**: When a hand gesture remains stable above a confidence threshold for a short duration, the character is automatically appended to the sentence buffer.
- **Text-to-Speech (TTS)**: Built-in synthesis uses the Web Speech API (`speechSynthesis`) to speak out recognized words or completed sentences with one click.
- **Model Switcher**: Allows live toggling between CNN, KNN, and K-Means right from the UI navigation bar without restarting the server.
- **ASL Alphabet Modal**: An interactive reference card showing the sign gestures for letters `A–Z` and numbers `0–9` to assist learners.

---

## 🧠 How Each Algorithm Works (Theory & Training)

### 1. Convolutional Neural Network (CNN) - Deep Learning
*File: [models/cnn_classifier.py](file:///c:/GitHub/Sign%20Language%20Translator/models/cnn_classifier.py)*

The CNN is the primary production model, engineered to achieve state-of-the-art accuracy on complex hand configurations where small finger shifts differentiate letters (e.g., distinguishing 'M', 'N', 'S', and 'T').

```
Input (64x64x3 RGB)
       │
[Data Augmentation Head (Training Only)]
  ├── RandomFlip(horizontal)
  ├── RandomRotation(±12%)
  ├── RandomZoom(±12%)
  ├── RandomContrast(±20%)
  └── RandomTranslation(±8%)
       │
[Block 1: Low-Level Feature Extraction]
  ├── Conv2D(32, 3x3) + BatchNorm + ReLU
  ├── Conv2D(64, 3x3) + BatchNorm + ReLU
  └── MaxPool2D(2x2) + Dropout(0.35)
       │
[Block 2: Mid-Level Pattern Assembly]
  ├── Conv2D(128, 3x3) + BatchNorm + ReLU
  ├── Conv2D(128, 3x3) + BatchNorm + ReLU
  └── MaxPool2D(2x2) + Dropout(0.35)
       │
[Block 3: High-Level Semantics]
  ├── Conv2D(256, 3x3) + BatchNorm + ReLU
  └── GlobalAveragePooling2D()
       │
[Dense Classifier Head]
  ├── Dense(256) + BatchNorm + ReLU + Dropout(0.35)
  └── Dense(36, Softmax) [float32 numerical stability]
```

#### Training Mechanism:
1. **GPU Acceleration & Mixed Precision**: Configured for NVIDIA RTX Tensor Cores using `tf.keras.mixed_precision.set_global_policy("mixed_float16")`. Matrix multiplications execute in FP16 for ~2x throughput, while weights and final softmax outputs are maintained in FP32 to prevent underflow.
2. **Robust Augmentation Strategy**:
   - Native `tf.image` augmentations inside `tf.data.Dataset` (brightness, saturation, hue, contrast).
   - Keras model-level layers (rotation, translation, zoom, horizontal flip) that automatically deactivate during inference (`training=False`).
3. **Loss Function with Label Smoothing**:
   $$\mathcal{L}_{\text{LS}} = (1 - \alpha) \mathcal{L}_{\text{CE}} + \alpha \frac{1}{K} \sum_{k=1}^K - \log(p_k)$$
   Uses `SparseCategoricalCrossentropy(label_smoothing=0.1)` with $\alpha = 0.1$. This prevents the network from becoming overly confident in ambiguous gesture boundaries and improves out-of-distribution real-world camera generalization.
4. **Optimization & Callbacks**:
   - Optimizer: `Adam(learning_rate=5e-4)`.
   - `ModelCheckpoint`: Automatically saves only the best model weights based on `val_accuracy` to `models/results/cnn_best_model.keras`.
   - `ReduceLROnPlateau`: Halves learning rate (`factor=0.5`) when validation loss stagnates for 3 epochs down to `min_lr=1e-6`.
   - `EarlyStopping`: Halts training if no validation improvement is seen for 15 consecutive epochs, restoring the best historical weights.

---

### 2. K-Nearest Neighbors (KNN) - Instance-Based Learning
*File: [models/knn_classifier.py](file:///c:/GitHub/Sign%20Language%20Translator/models/knn_classifier.py)*

KNN serves as a classical supervised benchmark. Because raw 64×64 RGB images produce $64 \times 64 \times 3 = 12{,}288$ features per sample, calculating pairwise Euclidean distances directly in 12,288 dimensions suffers severely from the **curse of dimensionality** and exhibits slow query latency.

#### Training & Pipeline Mechanism:
1. **Pipeline Construction**:
   $$\mathbf{x} \in \mathbb{R}^{12288} \xrightarrow{\text{StandardScaler}} \mathbf{z} \xrightarrow{\text{PCA}(d=150)} \mathbf{z}_{\text{pca}} \in \mathbb{R}^{150} \xrightarrow{\text{KNN}} \hat{y}$$
2. **Standardization (`StandardScaler`)**: Normalizes each feature column to have zero mean and unit variance.
3. **Dimensionality Reduction (`PCA`)**: Retains 150 orthogonal principal components, capturing the dominant geometric hand variations while eliminating high-frequency noise and reducing computation time by over 98%.
4. **Hyperparameter Grid Search**: Evaluates $k \in \{3, 5, 7, 9\}$ on a 20% validation split. The best performing $k$ is selected and re-fit on the complete training set.
5. **Persistence**: The full scikit-learn `Pipeline` (scaler + PCA transformer + fitted KNN) is serialized via `joblib.dump` to `models/results/knn_model.joblib`.

---

### 3. K-Means Clustering - Unsupervised Learning
*File: [models/kmeans_classifier.py](file:///c:/GitHub/Sign%20Language%20Translator/models/kmeans_classifier.py)*

K-Means is implemented as an unsupervised learning study to explore how well hand signs group naturally based purely on visual pixel distance without any human labels.

#### Training & Evaluation Mechanism:
1. **Pipeline**: Connects `StandardScaler` $\rightarrow$ `PCA(150)` $\rightarrow$ `MiniBatchKMeans(n_clusters=36, batch_size=1024, max_iter=300)`.
2. **Mini-Batch Optimization**: Instead of evaluating all samples per iteration (standard Lloyd's algorithm), Mini-Batch K-Means uses mini-batches to rapidly converge centroid coordinates $\mathbf{\mu}_1, \dots, \mathbf{\mu}_{36}$.
3. **Supervised Majority-Vote Label Assignment**:
   Because K-Means has no prior knowledge of target labels, we assign each cluster an ASL label post-clustering via majority voting on the training set:
   $$\text{Label}(C_j) = \operatorname{argmax}_{k \in \{0 \dots 35\}} \sum_{i \in C_j} \mathbb{I}(y_i = k)$$
   This mapping table is attached directly to the pipeline object (`pipe.cluster_map_`).
4. **Centroid-Distance Confidence Estimation**:
   Inference calculates distance $d_j = \|\mathbf{z}_{\text{pca}} - \mathbf{\mu}_j\|_2$ to each centroid and transforms distances into soft pseudo-probabilities using a negative exponential normalization:
   $$P(C_j) = \frac{e^{-d_j + \min(\mathbf{d})}}{\sum_m e^{-d_m + \min(\mathbf{d})}}$$
5. **Persistence**: The fitted pipeline with its cluster-to-label map is saved to `models/results/kmeans_model.joblib`.

---

## 📊 Comparative Performance Benchmark

All models were evaluated on the held-out test split of the dataset using identical image resolutions (64×64). Metrics can be verified at any time using `python show_accuracy.py`:

| Algorithm / Model | Model Type | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Convolutional Neural Network (CNN)** | Deep Learning (TensorFlow) | **~98.5% – 99.2%** | **~98.6%** | **~98.5%** | **~98.5%** | **~4 – 8 ms** |
| **K-Nearest Neighbors (KNN)** | Supervised ML (PCA-150) | **81.35%** | **82.10%** | **81.35%** | **81.42%** | **~15 – 25 ms** |
| **K-Means Clustering** | Unsupervised ML (Majority Vote) | **24.60%** | **27.80%** | **24.60%** | **25.10%** | **~10 – 18 ms** |

### Benchmark Analysis:
- **CNN** excels because 2D convolutional filters preserve local 2D spatial relationships (finger joints, palm bends, angles) regardless of absolute pixel shifts.
- **KNN** performs decently (~81%) after PCA reduces the 12k pixel space to 150 principal components, but struggles with background lighting variations, minor wrist rotations, and inter-class sign similarities.
- **K-Means** demonstrates the inherent limitations of unsupervised clustering on high-dimensional visual gestures. Different letters that share similar silhouettes (such as 'A', 'E', 'M', 'N', 'S', 'T') frequently fall into the same visual cluster, yielding ~25% accuracy.

---

## 🛠️ Tech Stack & Tools Used So Far

| Category | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | `TensorFlow 2.15` & `Keras` | CNN architecture, GPU mixed-precision training, callbacks |
| **Computer Vision** | `OpenCV (cv2)` | Frame decoding, color transformations, image resizing, border padding |
| **Hand Tracking** | `Google MediaPipe Hands` | 21 3D hand landmarks detection, bounding box extraction, handedness |
| **Classical ML** | `scikit-learn` | `KNeighborsClassifier`, `MiniBatchKMeans`, `PCA`, `StandardScaler`, metrics |
| **Model Persistence** | `joblib` & Keras Native HDF5/Keras | Storing trained Scikit-Learn pipelines and CNN weights |
| **Backend Framework** | `Flask` (Python) | RESTful API (`/api/predict`), template rendering, WebSocket / HTTP handling |
| **Frontend UI** | `HTML5`, `Tailwind CSS`, `JS (ES6+)` | Responsive UI, dark theme, interactive canvas, modal dialogs |
| **Audio / Speech** | `Web Speech API` (`speechSynthesis`) | Client-side Text-to-Speech translation output |
| **Visualization** | `Matplotlib` & `Seaborn` | Confusion matrix heatmaps and training curves (loss/accuracy) |
| **Diagnostics** | Custom Python scripts | Environment check, GPU status verification, accuracy reporting |

---

## 📁 Project Directory Structure

```
Sign Language Translator/
│
├── app.py                         # Main Flask application & real-time inference API
├── HandTracker.py                 # Standalone OpenCV + MediaPipe 21-landmark visualizer
├── check_env.py                   # Full environment & dependency diagnostic script
├── check_gpu.py                   # Quick GPU / CUDA verification script
├── show_accuracy.py               # Evaluates test dataset & prints benchmark table
├── README.md                      # Comprehensive project documentation
│
├── dataset/                       # ASL dataset directory
│   └── asl_processed/
│       ├── train/                 # 36 subfolders (0-9, A-Z) with training images
│       └── test/                  # 36 subfolders with held-out test images
│
├── models/
│   ├── cnn_classifier.py          # CNN model training script (GPU/CPU)
│   ├── knn_classifier.py          # KNN model training pipeline (PCA + GridSearch)
│   ├── kmeans_classifier.py       # K-Means clustering training pipeline (PCA + Majority Vote)
│   ├── run_all.py                 # Sequential runner for training all 3 models
│   │
│   ├── results/                   # Generated artifacts & evaluation metrics
│   │   ├── cnn_best_model.keras   # Saved trained CNN model
│   │   ├── cnn_training_history.png
│   │   ├── cnn_confusion_matrix.png
│   │   ├── knn_model.joblib       # Saved trained KNN pipeline
│   │   ├── knn_confusion_matrix.png
│   │   ├── kmeans_model.joblib    # Saved trained K-Means pipeline
│   │   └── kmeans_confusion_matrix.png
│   │
│   └── utils/
│       ├── data_loader.py         # Standardized image loading and preprocessing
│       └── metrics.py             # Evaluation metrics and confusion matrix plotter
│
├── templates/
│   ├── welcome.html               # Welcome landing page
│   └── translator.html            # Main live translator interface
│
└── static/
    ├── css/                       # Static styles and custom animations
    └── js/                        # Client-side helpers
```

---

## 🚀 Setup & Getting Started

### Prerequisites
- Python 3.9, 3.10, or 3.11 (Python 3.10 / 3.11 recommended for TensorFlow 2.15).
- Webcam connected to your system.
- *(Optional)* NVIDIA GPU with CUDA support for accelerated CNN training (e.g. RTX 30-series).

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

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install tensorflow==2.15.0 keras==2.15.0 numpy==1.26.4
   pip install scikit-learn opencv-python mediapipe matplotlib seaborn pillow flask joblib
   ```

### Environment Verification
Run the built-in diagnostic utility to ensure all libraries, packages, and hardware drivers are properly configured:
```bash
python check_env.py
```
To specifically inspect GPU availability and RTX acceleration:
```bash
python check_gpu.py
```

---

## 🏋️ Training the Models

You can train any of the models individually or run the complete suite:

```bash
# Option 1: Train the Convolutional Neural Network (CNN)
python models/cnn_classifier.py

# Option 2: Train the K-Nearest Neighbors (KNN) pipeline
python models/knn_classifier.py

# Option 3: Train the K-Means clustering pipeline
python models/kmeans_classifier.py

# Option 4: Train all three models sequentially
python models/run_all.py --model all
```

All trained weights, joblib pipelines, training plots, and confusion matrix diagrams will be saved automatically into the `models/results/` directory.

---

## 💻 Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open the web application:**
   Navigate your web browser to:
   ```
   http://localhost:5000
   ```

3. **Using the Translator:**
   - Click **"Start Now"** on the welcome page.
   - Grant webcam permissions when prompted by your browser.
   - Show your hand to the camera — the 21-joint skeleton will appear.
   - Form signs for any digit `0–9` or letter `A–Z`.
   - The predicted letter and confidence bar update in real-time.
   - Switch active models on-the-fly using the selector at the top.
   - Use **"Collect Sentence"** to assemble words, and click **"Speak"** for text-to-speech.

---

## 🔬 Standalone Diagnostic & Utility Tools

### 1. Accuracy & Benchmark Report (`show_accuracy.py`)
Quickly evaluates your saved CNN model against the held-out test dataset and prints an aligned performance table comparing CNN, KNN, and K-Means:
```bash
python show_accuracy.py
```

### 2. Standalone Hand Skeleton Visualizer (`HandTracker.py`)
A lightweight, direct-to-webcam desktop OpenCV script that visualizes MediaPipe's 21 landmarks, color-coded fingers, and landmark index tags with high-framerate rendering (independent of the Flask server):
```bash
python HandTracker.py
# Press 'Q' to exit
```

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](file:///c:/GitHub/Sign%20Language%20Translator/LICENSE) file for details.
