# 🧠 Deep Learning Portfolio & Experiments

Welcome to my Deep Learning repository! This repository showcases practical implementations of Computer Vision and Neural Network architectures built with **TensorFlow** and **Keras**. 

The goal of this project is to explore, compare, and demonstrate fundamental concepts in deep learning—ranging from basic Multi-Layer Perceptrons (MLP) to Convolutional Neural Networks (CNN).

---

## 📑 Projects Overview

| # | Project Name | Dataset | Architecture | Key Characteristics | Link |
|---|--------------|---------|--------------|---------------------|------|
| 01 | **Handwritten Digit Recognition** | MNIST | MLP (4 Hidden Layers) | Dense layers with ReLU, 97.10% Test Accuracy | [View Project](./01-mnist-classification) |
| 02 | **Image Classification (MLP)** | CIFAR-10 | MLP (3 Hidden Layers) | Fully connected baseline on RGB images | [View Project](./02-cifar10-mlp) |
| 03 | **Image Classification (CNN)** | CIFAR-10 | CNN (Conv2D + MaxPool) | Feature extraction via 2D Convolutions & Pooling | [View Project](./03-cifar10-cnn) |

---

## 📊 Key Insights & Architecture Comparison

### MNIST vs CIFAR-10 Complexity
- **MNIST:** Simple grayscale images (28x28x1). Fully Connected / MLP networks achieve high accuracy (>97%) with minimal training effort.
- **CIFAR-10:** Complex color images (32x32x3). Multi-Layer Perceptrons struggle to capture spatial hierarchies and spatial correlations in color images, leading to lower accuracy and potential overfitting.

### MLP vs CNN on CIFAR-10
- **Multi-Layer Perceptron (MLP):** Treats 3D images as flat 1D vectors (3072 features), ignoring local pixel spatial structures.
- **Convolutional Neural Network (CNN):** Uses weight sharing and localized kernels (`Conv2D` + `MaxPooling2D`) to preserve spatial context, achieving significantly better performance on complex visual data.

---

## 📂 Repository Structure

```text
deep-learning-projects/
│
├── README.md                      # Main repository documentation
│
├── 01-mnist-classification/       # Digit classification using MLP
│   ├── mnist_classifier.py
│   ├── requirements.txt
│   └── README.md
│
├── 02-cifar10-mlp/                # CIFAR-10 image classification using MLP
│   ├── cifar10_mlp.py
│   └── README.md
│
└── 03-cifar10-cnn/                # CIFAR-10 image classification using CNN
    ├── cifar10_cnn.py
    └── README.md
