# 🩺 Breast Cancer Classification using Deep Learning

This project builds a Convolutional Neural Network (CNN) called **CancerNet** to classify breast histology images as **cancerous** or **non-cancerous**.

---

## 📚 Project Overview

- **Problem**: Automatically detect Invasive Ductal Carcinoma (IDC) in microscopic breast tissue images.
- **Solution**: Train a CNN (CancerNet) using Keras and TensorFlow to classify images as either IDC-positive (1) or IDC-negative (0).
- **Dataset**: Dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images).

---

## 🚀 How the Project Works

### 1️⃣ Data Preparation (build_dataset.py)
Split images into:

- Training set (80%)
- Validation set (10% of training)
- Test set (20%)

Organize images into separate folders for:  
- 0: IDC-negative (benign)  
- 1: IDC-positive (malignant)

### 2️⃣ Model Design (cancernet.py)
- Built a custom CNN architecture named CancerNet:
- Separable Convolutional Layers to detect patterns.
- Batch Normalization to stabilize learning.
- Dropout Layers to prevent overfitting.

Output:
2 classes (0 or 1).

### 3️⃣ Model Training (train_model.py)
Data Augmentation using Keras' ImageDataGenerator:
- Rescaling, rotations, shifts, flips, zoom, shear.
- Optimizer: Adagrad.
- Loss Function: Binary Crossentropy.
- Handle class imbalance with class_weight.

Training includes:
- Monitoring validation accuracy.
- Saving a plot of training/validation loss and accuracy over time.

### 4️⃣ Model Evaluation
Evaluate model on unseen test images.

Report:
- Accuracy
- Sensitivity (Recall) – Ability to detect cancer.
- Specificity – Ability to detect non-cancer.
- Confusion Matrix for detailed performance.

## 🛠️ Skills Used

### 📌 Deep Learning & Neural Networks
- Built and trained a custom **Convolutional Neural Network (CNN)** for image classification.
- Applied **Separable Convolution** layers for efficient computation and reduced overfitting.
- Used **Batch Normalization** and **Dropout** to improve model generalization and stability.
- Designed and implemented the CNN using **Keras** with a **TensorFlow backend**.

### 📌 Data Handling & Processing
- Managed large-scale image datasets (275k+ images).
- Automated **dataset splitting** into training, validation, and test sets using Python scripting.
- Applied **Image Augmentation** (rotation, shifting, flipping, zoom, shear) using Keras’ `ImageDataGenerator` to enhance training robustness.
- Balanced class imbalance with **class weighting** during training.

### 📌 Model Training & Optimization
- Trained the CNN using **Adagrad Optimizer** with a custom learning rate schedule.
- Monitored performance via **validation loss/accuracy**.

### 📌 Performance Evaluation
- Generated and interpreted:
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-score)
  - **Sensitivity (Recall)** and **Specificity**
  - **Overall Accuracy**

### 📌 Visualization & Reporting
- Created and saved training/validation **loss and accuracy plots** using **Matplotlib**.
- Summarized and interpreted model performance in readable reports.

### 📌 Software Engineering Practices
- Modularized code into:
  - Data preparation (`build_dataset.py`)
  - Model architecture (`cancernet.py`)
  - Training and evaluation (`train_model.py`)
- Applied good file organization and configuration management (`config.py`).
- Used libraries such as:
  - `NumPy` and `scikit-learn` for data handling and evaluation
  - `imutils` for file path management
  - `OpenCV` and `Pillow` for image processing

---

## 🚀 Tools & Libraries

- **Python**
- **TensorFlow** (with **Keras** API)
- **NumPy**
- **scikit-learn**
- **OpenCV**
- **Pillow**
- **Matplotlib**
- **imutils**

---

## Credits
Created by Louis Nguyen
