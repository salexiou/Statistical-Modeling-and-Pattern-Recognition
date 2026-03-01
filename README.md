# 📈 Statistical Modeling & Pattern Recognition

A comprehensive collection of machine learning algorithms and statistical models implemented from scratch (without utilizing high-level ML libraries). 

Developed for the **"Statistical Modeling and Pattern Recognition" (THL311)** course at the Technical University of Crete (ECE Department).

## 🚀 Project Overview

This repository contains the implementation and analytical evaluation of foundational dimensionality reduction techniques, feature extraction methods, and statistical classifiers. The project spans across various datasets, moving from mathematical proofs to practical programming applications in both **MATLAB/Octave** and **Python**.

## 🧠 Core Topics & Implementations

### 1. Principal Component Analysis (PCA)
* **Custom Implementation:** Built PCA from scratch using covariance matrices and eigendecomposition.
* **Datasets:** Breast Cancer dataset (569 samples, 30 features) and a Faces dataset (5,000 images).
* **Tasks:** Data standardization, dimensionality reduction (e.g., 2D to 1D, or compressing high-dimensional face images), 2D/3D visualizations, data recovery/reconstruction, and explained variance analysis.

### 2. Linear Discriminant Analysis (LDA)
* **Mathematical & Programmatic Application:** Calculated within-class ($S_w$) and between-class ($S_b$) scatter matrices to find the optimal projection vectors.
* **Datasets:** Synthetic 2D data and the classic **Fisher's Iris dataset**.
* **Comparison:** Contrasted LDA's class-separability performance against PCA's variance-maximization on the same datasets.

### 3. Bayesian Classification & Decision Boundaries
* **Gaussian Distributions:** Plotted 2D contour lines for bivariate normal distributions.
* **Decision Boundaries:** Calculated and visualized decision boundaries between two classes, analyzing how varying *a-priori* probabilities ($P(\omega_1)$) shift the boundaries.
* **Minimum Risk:** Analytically solved for the optimal decision boundary $x_0$ using a Rayleigh distribution and a predefined cost/risk matrix.

### 4. Feature Extraction & Naive Bayes on MNIST
* **Dataset:** A subset of the **MNIST** handwritten digits dataset (digits 0, 1, and 2).
* **Feature Engineering:** Extracted custom features from raw pixel data using Python:
  * **Aspect Ratio:** Bounding box width/height ratio.
  * **Foreground Pixels:** Count of non-zero pixels.
  * **Centroid:** Center of mass of the digit mapped to a 1D feature.
* **Classification:** Trained a custom Naive Bayes classifier assuming feature independence and evaluated its accuracy across different digit class combinations.

## 💻 Tech Stack
* **Languages:** MATLAB / Octave, Python 3
* **Libraries (Python):** `numpy`, `matplotlib`, `pandas` (for data loading and matrix operations only; ML algorithms were written from scratch).
* **Concepts:** Dimensionality Reduction, Statistical Classification, Feature Extraction, Eigendecomposition, Minimum Risk Theory.
