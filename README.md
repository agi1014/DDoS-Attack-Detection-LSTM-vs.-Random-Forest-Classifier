**DDoS Attack Detection: LSTM vs. Random Forest Classifier**
**Project Overview**
This repository contains a Jupyter Notebook (DDoS_Detection_Analysis.ipynb) that conducts a detailed comparative study of machine learning models for detecting Distributed Denial of Service (DDoS) attacks. Leveraging a Cleaned_DDoS_Dataset.csv (assumed to be provided separately), the project implements a robust machine learning pipeline designed to handle challenges common in cybersecurity datasets, such as class imbalance and data imperfections.
The core objective is to evaluate and compare the effectiveness of two distinct machine learning paradigms: a Long Short-Term Memory (LSTM) neural network (for sequential pattern recognition) and a Random Forest Classifier (a powerful ensemble tree-based method) in accurately identifying malicious DDoS traffic.
This notebook serves as a valuable resource for anyone interested in network security, anomaly detection, and applying deep learning/ensemble methods to real-world cybersecurity problems.

**Key Features & Methodology**

**Data Loading & Initial Preparation:**
Loads Cleaned_DDoS_Dataset.csv.
Standardizes column names and renames the target variable to label.

**Advanced Data Balancing & Augmentation:**
Addresses severe class imbalance (DDoS vs. Normal traffic) by generating synthetic 'Normal' traffic samples. This is achieved by taking existing DDoS samples, re-labeling them as normal, and applying Gaussian noise, multiplicative scaling, and offset to create realistic variations, moving beyond simple oversampling.

**Controlled Label Noise Injection:**
To evaluate model robustness, varying levels of controlled label noise are intentionally introduced into the balanced datasets:
5% label noise for the LSTM model's dataset.
12% label noise for the Random Forest model's dataset (simulating more challenging scenarios).

**Dynamic Feature Selection:**

A subset of features (25% of the total, with a minimum of 3) is randomly selected. This helps in understanding model performance under potentially resource-constrained or feature-limited environments.

**Data Normalization:**

**MinMaxScaler** is applied to all selected features, an essential preprocessing step for neural networks and beneficial for many other ML algorithms.

**Model Training & Evaluation:**

**LSTM Model:**

A sequential Keras model with LSTM layers, BatchNormalization, and Dropout for regularization.

Compiled with the Adam optimizer and BinaryFocalCrossentropy loss, specifically chosen for its effectiveness in imbalanced classification tasks.

Trained with EarlyStopping to prevent overfitting.

**Random Forest Classifier:**

Configured with n_estimators=150 and max_depth=10 for robust performance.

Utilizes class_weight='balanced' to effectively handle the adjusted class distributions.

Comprehensive Performance Metrics & Visualizations:

Classification Report: Precision, Recall, F1-Score, Support for both classes.

Accuracy Score: Overall prediction accuracy.

ROC AUC Score: Area Under the Receiver Operating Characteristic Curve.

Confusion Matrix: Visual representation of true positives, true negatives, false positives, and false negatives.

ROC Curve: Plot of True Positive Rate vs. False Positive Rate.

Precision-Recall Curve: Particularly useful for imbalanced datasets.

Accuracy vs. Threshold Plot: Illustrates how accuracy changes with different classification probability thresholds.

Recall & F1 Score vs. Threshold Plots (per class): Detailed analysis of threshold impact on specific class performance.

Feature Importance Plot: Visualizes the most influential features identified by the Random Forest model.

Model Accuracy Comparison Plot: A direct visual comparison of the final accuracies of the LSTM and Random Forest models.

**Results & Insights**
The notebook provides a clear side-by-side comparison, highlighting that the LSTM model achieved a higher overall accuracy (approx. 94.17%) and AUC (approx. 0.7592) compared to the Random Forest model (approx. 87.73% accuracy, 0.6489 AUC) in the tested noisy environment. This suggests the LSTM's ability to learn temporal dependencies even with introduced noise, which can be crucial for detecting evolving attack patterns. The detailed visualizations offer a deeper understanding of each model's strengths and weaknesses regarding false positives/negatives, crucial for sensitive applications like DDoS detection.

**Requirements**

pandas

numpy

matplotlib

seaborn

scikit-learn

tensorflow

**Bash**

**pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
Usage**

Place your dataset: Ensure your dataset, named Cleaned_DDoS_Dataset.csv, is located in the root directory of this project.

Run the notebook: Open DDoS_Detection_Analysis.ipynb using Jupyter Lab, Jupyter Notebook, or a compatible IDE (like VS Code with the Python extension) and run all cells sequentially.
