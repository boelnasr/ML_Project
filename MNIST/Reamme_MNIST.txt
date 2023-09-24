# MNIST Classification using KNN and CNN

This project involves the classification of the MNIST dataset using two distinct methodologies: k-Nearest Neighbors (KNN) and Convolutional Neural Networks (CNN). The MNIST dataset, a staple in the machine learning community, consists of handwritten digits and has served as a benchmark for classification algorithms for many years.

## 📜 Scripts Overview

### 1. `mnist_KNN.m`

This script undertakes the task of classifying the MNIST dataset using the k-Nearest Neighbors algorithm. After the application of a dimensionality reduction technique (PCA), the KNN algorithm gets to work on the task of classifying the handwritten digits.

**Key Features**:
- **Data Handling**: Load and preprocess the data.
- **PCA**: Implement Principal Component Analysis for dimensionality reduction.
- **KNN Algorithm**: Train and test the data set.
- **Performance Metrics**: Evaluate based on accuracy, precision, recall, and F1-score.

### 2. `mnist_cnn.m`

This script leverages a Convolutional Neural Network to classify the MNIST dataset.

**Key Features**:
- **Data Handling**: Load and preprocess the data.
- **CNN Architecture**: Define the structure using MATLAB's `layerGraph`.
- **CNN Training**: Set parameters and configurations for training.
- **Performance Metrics**: Evaluate based on a confusion matrix, accuracy, precision, recall, and F1-score.

## 🚀 Getting Started

1. **Clone** this repository to your local machine.
2. Ensure you have **MATLAB** installed with the requisite toolboxes (**Deep Learning Toolbox** for the CNN, **Statistics and Machine Learning Toolbox** for KNN).
3. **Navigate** to the project directory and **execute** the desired script (`mnist_KNN.m` or `mnist_cnn.m`).
4. **make sure** mnist dataset is in the same directory 

## 📊 Results

Upon script execution, you'll see the results, inclusive of the confusion matrix, accuracy, precision, recall, and F1-score. For an intricate comparison between the two methodologies, please refer to the companion LaTeX document.

## 🔧 Requirements

- MATLAB (version utilized for development: R2022a)
- Deep Learning Toolbox (essential for the CNN approach)
- Statistics and Machine Learning Toolbox (vital for the KNN approach)

