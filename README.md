# ✋ Hand Gesture Recognition using MediaPipe and Random Forest

This project aims to recognize hand gestures using **MediaPipe** for landmark detection and **Random Forest** for gesture classification.

## 🎯 Objective

To build a simple and effective hand gesture recognition system based on hand landmarks extracted using MediaPipe and classify them using a machine learning model.

## 📷 Input

- Live video or images (via webcam)
- MediaPipe extracts 21 hand landmarks per frame (each with x, y, z coordinates)

## 🧠 Feature Extraction

- Each hand gives 21 landmarks.
- Each landmark has 3 coordinates: (x, y, z).
- Final input vector: **63 features** per frame (21 × 3).

## 📊 Model

- **Algorithm**: Random Forest Classifier
- **Input**: Normalized 63-dimensional vector (21 landmarks × 3 coordinates)
- **Output**: Gesture class label
- **Note**: Hyperparameters of the Random Forest have **not been optimized** yet, but the model performs **decently well** on the current dataset.

## ⚙️ Dependencies

- `mediapipe`
- `scikit-learn`
- `opencv-python`
- `numpy`
- `pandas`

Install with:

```bash
pip install mediapipe scikit-learn opencv-python numpy pandas
