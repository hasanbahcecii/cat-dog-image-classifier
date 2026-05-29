# 🐾 Cat vs Dog Image Classifier — Transfer Learning with MobileNetV2

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
├── cnn.py                  # Original from-scratch CNN (baseline)
├── train_transfer.py       # Transfer learning model (MobileNetV2)
│
├── model_final.keras       # Saved best model
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
python cnn.py
```

**Single prediction:**
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('model_final.keras')
img = image.load_img('your_image.jpg', target_size=(224, 224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)
prob = model.predict(x, verbose=0)[0][0]
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
| Augmentation | Keras ImageDataGenerator |
| Regularization | Dropout, EarlyStopping |
| Optimizer | Adam (Phase 1) / Adam lr=1e-5 (Phase 2) |

---

## 📜 License

MIT License — open source, free to use and modify.