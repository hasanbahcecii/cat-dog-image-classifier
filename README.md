# 🧠 CNN Image Classification with TensorFlow

This repository contains a **Deep Learning project** that implements a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify images of **cats 🐱 and dogs 🐶**.

---

## 📌 Project Overview

The goal of this project is to build a CNN that can distinguish between cat and dog images.

The model is trained on labeled image data and learns spatial features using convolutional layers.

---

## 🗂 Dataset

Download the dataset from the link below:

📦 [Download dataset.zip](https://www.dropbox.com/scl/fi/ppd8g3d6yoy5gbn960fso/dataset.zip?rlkey=lqbqx7z6i9hp61l6g731wgp4v&e=1&st=gdn6pydw&dl=0)

After extracting, the folder structure should be:
```
dataset/
│
├── training_set/
│ ├── cats/
│ └── dogs/
│
└── test_set/
├── cats/
└── dogs/
```
---

### 🚀 CNN Workflow

1. **Preprocessing** the training and test image sets
2. **Building** a CNN with:
   - 2 convolution + pooling layers
   - 1 fully connected layer
3. **Training** on 25 epochs
4. **Predicting** a single image
   
---

## ⚙️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  

---

## 🙋‍♂️ Acknowledgments

- Dataset for CNN provided via Dropbox link

- Projects built with educational and demonstration purposes in mind

---

## License
This project is licensed under the [MIT License](https://opensource.org/license/MIT).