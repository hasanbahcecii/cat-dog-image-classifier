# 🐾 Cat vs Dog Image Classifier — Transfer Learning with MobileNetV2

🌐 **[Live Demo — Streamlit](https://hasanbahcecii-cat-dog-image-classifier-app-gtgtuq.streamlit.app)**

A binary image classification system built with TensorFlow/Keras, using **transfer learning from MobileNetV2** (ImageNet pre-trained) with 2-phase training and data augmentation.

---

## 📊 Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **98.70%** |
| Test Loss | 0.0391 |
| Best Val Accuracy | 98.85% |
| Architecture | MobileNetV2 + Custom Head |
| Training Phases | 2 (Feature Extraction → Fine-Tuning) |
| Dataset | 8,000 train / 2,000 test images |

---

## 🏗️ Architecture

```
Input (224×224×3)
        ↓
MobileNetV2 (ImageNet pre-trained, frozen in Phase 1)
        ↓
GlobalAveragePooling2D
        ↓
Dropout(0.3)
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.2)
        ↓
Dense(1, Sigmoid) → cat / dog
```

### 2-Phase Training Strategy

**Phase 1 — Feature Extraction**
- MobileNetV2 base frozen
- Only custom head trained
- Adam optimizer, lr=1e-3
- EarlyStopping (patience=3)

**Phase 2 — Fine-Tuning**
- Top 30 layers of MobileNetV2 unfrozen
- Very low learning rate (lr=1e-5) to avoid catastrophic forgetting
- EarlyStopping (patience=3)

---

## 📂 Project Structure

```
cat-dog-image-classifier-cnn/
│
├── convolutional_neural_network.py   # Baseline CNN (from scratch)
├── train_transfer.py                 # Transfer learning — MobileNetV2
├── app.py                            # Streamlit web interface
│
├── model_final.keras                 # Saved Keras model
├── model_final.onnx                  # ONNX export (used by Streamlit app)
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

📦 [Download dataset.zip](https://www.dropbox.com/scl/fi/ppd8g3d6yoy5gbn960fso/dataset.zip?rlkey=lqbqx7z6i9hp61l6g731wgp4v&e=1&st=gdn6pydw&dl=0)

```
dataset/
│
├── training_set/
│   ├── cats/       # 4,000 images
│   └── dogs/       # 4,000 images
│
└── test_set/
    ├── cats/       # 1,000 images
    └── dogs/       # 1,000 images
```

---

## ⚙️ Installation

```bash
git clone https://github.com/hasanbahcecii/cat-dog-image-classifier-cnn
cd cat-dog-image-classifier-cnn

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Usage

**Train transfer learning model:**
```bash
python train_transfer.py
```

**Run baseline CNN (from scratch):**
```bash
python convolutional_neural_network.py
```

**Run Streamlit web interface (local):**
```bash
streamlit run app.py
```

**Single prediction:**
```python
import numpy as np
import onnxruntime as ort
from PIL import Image

sess = ort.InferenceSession('model_final.onnx')
input_name = sess.get_inputs()[0].name

img = Image.open('your_image.jpg').convert('RGB').resize((224, 224))
x = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
prob = sess.run(None, {input_name: x})[0][0][0]
print("dog" if prob > 0.5 else "cat", f"({max(prob, 1-prob)*100:.1f}%)")
```

---

## 🔄 Augmentation Pipeline

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

---

## 🛠️ Technologies

| Component | Technology |
|---|---|
| Framework | TensorFlow / Keras |
| Base Model | MobileNetV2 (ImageNet) |
| Inference (deploy) | ONNX Runtime |
| Web Interface | Streamlit |
| Augmentation | Keras ImageDataGenerator |
| Regularization | Dropout, EarlyStopping |
| Optimizer | Adam (Phase 1) / Adam lr=1e-5 (Phase 2) |

---

## 📜 License

MIT License — open source, free to use and modify.