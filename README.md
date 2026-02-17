

# 📄 `README.md`

```markdown
# 🌀 Naruto Hand Sign Recognition (Real-Time)

A real-time Naruto hand seal recognition system built using  
MediaPipe + Landmark-Based ML + Scikit-Learn.

This project dynamically tracks single and two-hand seals and predicts them in real time using a trained neural network model.

---

## 🚀 Features

- ✅ Real-time hand tracking (CPU friendly)
- ✅ Single & two-hand seal support
- ✅ 126-dimension landmark feature vector
- ✅ Wrist-relative normalization
- ✅ Data augmentation (flip, rotate, brightness, zoom)
- ✅ MLP-based classifier
- ✅ Dynamic prediction smoothing
- ✅ Stable live recognition
- ✅ Ready for WebSocket & Three.js integration

---

## 🧠 How It Works

1. MediaPipe detects hand landmarks (21 points per hand)
2. Landmarks are normalized relative to wrist
3. Left + Right hands are merged into a 126-dim vector
4. Model predicts the Naruto seal
5. Temporal smoothing ensures stable output

Pipeline:

```

Webcam → MediaPipe → Normalize → MLP Model → Smooth Output → Display

```

---

## 📁 Project Structure

```

naruto-hand-sign-recognition/
│
├── dataset/
│   └── test/
│       ├── bird/
│       ├── boar/
│       ├── dog/
│       ├── horse/
│       ├── monkey/
│       ├── ram/
│       ├── snake/
│       └── tiger/
│
├── dataset_webcam/        # Optional real webcam samples
│
├── models/
│   └── naruto_seal_model.pkl
│
├── src/
│   ├── train_model.py
│   ├── live_dynamic_recognition.py
│   ├── webcam_dataset_collector.py
│   └── hand_skeleton_test.py
│
├── requirements.txt
└── README.md

```

---

## ⚙️ Installation

### 1️⃣ Install Python (Recommended: 3.10)

Check version:

```

python --version

```

### 2️⃣ Install Dependencies

```

pip install -r requirements.txt

```

---

## 🏋️ Training the Model

Make sure your dataset is placed in:

```

dataset/test/<seal_name>/*.jpg

```

Then run:

```

python src/train_model.py

```

Output includes:

- Training accuracy
- Validation accuracy
- Confusion matrix
- Model saved to `models/`

---

## 🎥 Real-Time Dynamic Recognition

Run:

```

python src/live_dynamic_recognition.py

```

Features:

- Continuous tracking
- Majority vote smoothing
- Stable seal display
- Works for single & two-hand poses

---

## 📸 Add Real Webcam Samples (Optional)

To improve accuracy:

```

python src/webcam_dataset_collector.py

```

- Press `S` to save sample
- Press `N` to switch seal
- Press `Q` to quit

Then retrain.

---

## 📊 Model Details

- Architecture: MLP (256 → 128 → 64)
- Activation: ReLU
- Optimizer: Adam
- Epochs: 350
- Feature Size: 126 (Left 63 + Right 63)
- Normalization: Wrist-relative scaling
- Validation Accuracy: ~92–95%

---

## 💡 Performance Notes

- Runs entirely on CPU
- No GPU required
- Real-time capable (~20–30 FPS)
- Accuracy improves with real webcam samples

---

## 🔮 Next Steps

- WebSocket backend integration
- Three.js jutsu animation triggering
- Confidence-based animation control
- Temporal seal detection sequences
- Model export for web deployment

---

## 🧩 Tech Stack

- Python
- OpenCV
- MediaPipe
- Scikit-Learn
- NumPy
- Joblib

---

## 📜 License

This project is for educational and experimental purposes.

Naruto and related content belong to their respective copyright owners.

---

## 👨‍💻 Author

Naruto Hand Sign Recognition  
Built with passion for Computer Vision & Anime ⚡
```

---


Just tell me.
