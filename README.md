# Diabetes Prediction using Machine Learning in Python 
## 📌 Project Overview
##### This project focuses on building a machine learning model that predicts whether a person has diabetes based on health-related features. Using the Pima Indians Diabetes Dataset from Kaggle, the project applies data preprocessing, exploratory data analysis, and classification algorithms, SVM, to predict diabetes outcomes. The workflow includes splitting the dataset into training and testing sets, standardizing features for better model performance, evaluating accuracy, and developing a simple predictive system to classify patient data. This project demonstrates a complete end-to-end process of building a supervised machine learning model for real-world medical predictions.
## 🎯 Objectives

##### Diabetes Classification – Build an SVM model to predict whether a person is diabetic.
##### Data Exploration & Preprocessing – Analyze the dataset, scale features, and prepare the data for SVM training.
##### Model Development – Train an SVM classifier with a linear kernel.
##### Performance Evaluation – Evaluate the model using metrics like accuracy, confusion matrix, and classification report.
##### Predictive System Creation – Develop a system that takes health features as input and outputs whether the person has diabetes.

## 📁 Dataset
##### Name: Pima Indians Diabetes Dataset
##### Source: Kaggle
##### Shape: 768 samples × 8 features
##### Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
##### Target Label:
##### 0 → Non-diabetic
##### 1 → Diabetic

## ⚙️ Methodology

### 1. Data Preprocessing
##### Loaded the dataset and checked for missing values.
##### Standardized all features to ensure equal contribution to the SVM model.
##### Separated features (X) and labels (Y) for model training.

### 2. Exploratory Data Analysis (EDA)
##### Reviewed dataset structure, feature distributions, and class balance.
##### Visualized correlations and key patterns to understand feature behavior.

## 3. Train-Test Split
##### Split the data into training and testing sets to evaluate model performance.

## 4. Model Development (SVM)
##### Trained an SVM classifier with a linear kernel to classify patients as diabetic or non-diabetic.
##### Fitted the model using X_train and Y_train.

## 5. Model Evaluation
##### Calculated performance metrics:
##### Training Accuracy: 78.66 %
##### Testing Accuracy: 77.273 %
##### Visualized results using a confusion matrix to show prediction performance.

## 6. Predictive System Creation
##### Developed a simple predictive system where new patient data (8 features) can be input to predict diabetes status (0 → Non-diabetic, 1 → Diabetic).

## 🚀 Results & Insights
##### Model Accuracy:
##### Training Accuracy: 78.66 %
##### Testing Accuracy: 77.272 %
##### Prediction System: Successfully created a real-time prediction system for new patient data.
##### Model Reliability: The gap between training and testing accuracy is small, indicating good generalization.
##### Key Insight: Features such as glucose level, BMI, and insulin contain enough information for a linear SVM to accurately predict diabetes risk.

## 🛠 Tech Stack
##### Python: Main programming language
##### NumPy: For numerical operations and arrays
##### Pandas: For loading, cleaning, and preprocessing the dataset
##### Scikit-learn: For building, training, and evaluating the SVM model
##### Matplotlib / Seaborn: For visualizations, including confusion matrix and feature distributions



