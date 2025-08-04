# Leukemia Cell Classification using ResNet50

This repository contains the code and resources for a deep learning project focused on classifying leukemia from microscopic blood cell images. The model leverages a pre-trained ResNet50 architecture to accurately distinguish between healthy cells and cells affected by Acute Lymphoblastic Leukemia (ALL).

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Overview

Acute Lymphoblastic Leukemia (ALL) is a cancer of the blood and bone marrow that requires prompt and accurate diagnosis for effective treatment. This project automates the detection process using computer vision and deep learning.

By fine-tuning a **ResNet50** model, which has been pre-trained on the extensive ImageNet dataset, we can apply powerful feature extraction capabilities to the specific task of medical image analysis. The model is trained to serve as an efficient, automated tool to assist hematologists and medical professionals in diagnosing leukemia.

### Key Features:
-   **Deep Learning Model:** Utilizes the ResNet50 Convolutional Neural Network (CNN).
-   **Transfer Learning:** Employs a model pre-trained on ImageNet for enhanced performance and faster convergence.
-   **Image Preprocessing:** Includes a robust pipeline with cell segmentation and image enhancement to improve classification accuracy.
-   **High Performance:** Achieves high accuracy on a held-out test set, demonstrating its effectiveness.

---

## 💾 Dataset

This project uses the **C-NMC_Leukemia dataset**, which is publicly available on Kaggle.

-   **Source:** [C-NMC_Leukemia Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)
-   **Content:** The dataset contains 10,661 microscopic images of blood cells.
-   **Classes:**
    -   `all` (Acute Lymphoblastic Leukemia)
    -   `hem` (Healthy)

The data is structured into three folds for training, which are combined for this project.

---

## 🛠️ Methodology

The project follows a standard machine learning workflow:

1.  **Image Preprocessing:**
    -   Images are loaded using OpenCV.
    -   **Cell Segmentation:** To isolate the cell of interest, a mask is created by converting the image to grayscale, applying Contrast Limited Adaptive Histogram Equalization (CLAHE), and using Otsu's thresholding.
    -   **Cropping:** The original image is cropped based on the segmented mask to remove irrelevant background noise.
    -   **Resizing:** All cropped images are resized to `224x224` pixels to match the input dimensions of the ResNet50 model.

2.  **Model Architecture:**
    -   The **ResNet50** model is used as the base, with its final classification layer removed.
    -   A new custom classifier head is added on top, consisting of `GlobalAveragePooling2D`, `Dropout` for regularization, and `Dense` layers.
    -   A `sigmoid` activation function is used in the final layer for binary classification.

3.  **Training and Evaluation:**
    -   The model is compiled with the Adam optimizer and `binary_crossentropy` loss.
    -   `EarlyStopping` and `ModelCheckpoint` callbacks are used to prevent overfitting and save the best model weights.
    -   The model is trained on 80% of the data and evaluated on the remaining 20% to measure its real-world performance.

---

## 📊 Results

The trained model demonstrates strong performance in classifying leukemia cells.

-   **Test Accuracy:** **89.87%**

The training history shows that the model learns effectively, with validation accuracy peaking early. `EarlyStopping` ensures the final model is both accurate and well-generalized.

![Training History Plot](plots/training_history.png)
*(This assumes you have a `plots` directory with the generated graph)*

---

## 📂 Repository Structure

The project is organized to promote clarity and reproducibility.
