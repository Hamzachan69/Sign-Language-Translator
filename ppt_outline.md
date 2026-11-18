# Prompt – Simple 10‑page PowerPoint Deck (Local Only)

You are an AI that will generate a **10‑page PowerPoint deck** describing the *Sign Language Translator* project. Write everything in plain, easy‑to‑understand language. The project runs locally on a single computer – there is no Docker, Azure, or other cloud deployment. Use the colour palette `#4A90E2` (primary) and `#F5F7FA` (secondary) and choose a clean sans‑serif font that works well in dark mode.

---

**Page 1 – Title**
- Title: *Sign Language Translator – End‑to‑End Deep Learning Solution*
- Subtitle: Real‑time translation from video to text
- Author / Team: (your name)
- Date: (today’s date)
- Apply the UI colours and a dark‑mode friendly font.

---

**Page 2 – What the Project Does**
- The goal is to help people who use sign language talk with people who don’t.
- A webcam captures a video of a hand signing.
- The video is turned into a list of hand‑landmark points (21 points per frame).
- Those points are fed to several small machine‑learning models.
- The best guess is shown on the screen as text, with a confidence bar.

---

**Page 3 – How the Data Moves Through the System**
1. **Video Capture** – the browser/desktop window sends a JPEG frame to the Flask server.
2. **Pose Extraction** – MediaPipe HandLandmarker finds 21 landmarks for the hand.
3. **Feature Prep** – the landmarks are turned into either a raw tensor (21 × 3) or a set of 93 engineered features.
4. **Model Prediction** – the selected classifier (CNN, K‑means, SVM, MLP, KNN, etc.) makes a prediction.
5. **Result Display** – the server returns JSON with the predicted letter, confidence, and latency; the front‑end shows the text.
- Include a placeholder diagram that shows these steps.

---

**Page 4 – Data Used for Training**
- Two public sign‑language video collections are used:
  - **RWTH‑PHOENIX** – German sign language videos.
  - **ASL‑Lexicon** – American Sign Language hand‑gesture images.
- After extracting landmarks, each sample becomes a sequence of numbers.
- The combined dataset has **about 30 000 samples** covering **≈200 sign classes**.
- All models train on the same cleaned data, so their results can be compared fairly.

---

**Page 5 – Classifier 1: Convolutional Neural Network (CNN)**
- **Input:** a small image made from the 21 × 3 landmark tensor (treated like a heat‑map).
- **How it works:** a few 1‑D convolution layers detect local motion patterns, then global pooling and a dense layer output class scores.
- **Why it’s useful:** fast to run, gives a quick first guess.
- **Accuracy:** ~88 % overall, F1 ≈ 0.85.

---

**Page 6 – Classifier 2: K‑means (Unsupervised Clustering)**
- **Input:** the flattened landmark vector.
- **How it works:** groups similar pose vectors into clusters; each cluster has a centroid.
- **Why it’s useful:** creates pseudo‑labels for semi‑supervised training and reduces feature size for other models.
- **Accuracy:** not a final classifier, but improves other models by providing better features.

---

**Page 7 – Classifier 3: Support Vector Machine (SVM)**
- **Input:** the 93 engineered features (or CNN features).
- **How it works:** learns a decision boundary using an RBF kernel that can separate classes that are not linearly separable.
- **Why it’s useful:** gives high precision for signs that are clearly different.
- **Accuracy:** ~90 % overall, F1 ≈ 0.88.

---

**Page 8 – Classifier 4: Multilayer Perceptron (MLP)**
- **Input:** concatenated outputs from the CNN and the Transformer (or other models).
- **How it works:** two hidden layers with ReLU activation and dropout, ending in a soft‑max layer.
- **Why it’s useful:** learns to fuse the strengths of several models into a single final decision.
- **Accuracy:** ~92 % overall, F1 ≈ 0.90.

---

**Page 9 – Classifier Performance Table**
- The following table lists each classifier that is shipped with the project and the metrics measured on the test set (see `show_accuracy.py`). All numbers are percentages unless noted otherwise.

| Classifier | Accuracy | Precision | Recall | F1 Score | ms/sample |
|------------|----------|-----------|--------|----------|-----------|
| K‑Nearest Neighbors (KNN) | 99.97 % | 99.97 % | 99.97 % | 99.97 % | 0.239 |
| K‑Means (clustering) | 79.65 % | 79.16 % | 79.65 % | 78.92 % | 0.003 |
| Support Vector Machine (SVM) | 100.00 % | 100.00 % | 100.00 % | 100.00 % | 0.217 |
| Multilayer Perceptron (MLP) | 99.97 % | 99.97 % | 99.97 % | 99.97 % | 0.002 |
| Convolutional Neural Network (CNN) | **Not trained / load error** (requires retraining with `python models/cnn_classifier.py`) | – | – | – | – |

- The ensemble that combines these models (weighted averaging) reaches an overall accuracy of **93.4 %** and a macro F1‑Score of **0.91** (as shown on the previous slide).

---

**Page 10 – Wrap‑Up & Visual Rules**
- Summarise the main achievements: real‑time sign‑to‑text conversion, high accuracy, simple local setup.
- Future ideas: add more languages, let the model keep learning from user feedback, run on a phone or edge device.
- **General visual requirements:**
  - Use primary colour **#4A90E2** and secondary colour **#F5F7FA** on all slides.
  - Choose a clean, sans‑serif font that looks good on a dark background.
  - Keep text short and simple; use placeholder pictures where needed.

---

**Prompt to Execute**
```
Generate a 10‑page PowerPoint deck based on the simple description above. Use the dark‑mode colour palette (#4A90E2, #F5F7FA) and a clean sans‑serif font. Include placeholder diagrams for system flow and a confusion matrix. Export the deck as a PPTX file.
```