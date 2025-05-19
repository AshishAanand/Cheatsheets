# 🧠 Supervised Learning Terminologies Cheat Sheet

> A comprehensive and beginner-friendly cheat sheet covering **core concepts**, **key algorithms**, and **important terms** used in **Supervised Machine Learning**.

---

## 📌 Table of Contents

* [What is Supervised Learning?](#what-is-supervised-learning)
* [Key Concepts](#key-concepts)
* [Types of Supervised Learning](#types-of-supervised-learning)
* [Common Algorithms](#common-algorithms)
* [Important Terminologies](#important-terminologies)
* [Model Evaluation Metrics](#model-evaluation-metrics)
* [Training vs Testing](#training-vs-testing)
* [Tips & Best Practices](#tips--best-practices)

---

## 🧭 What is Supervised Learning?

Supervised learning is a type of **Machine Learning** where the model is trained on a **labeled dataset**.

🔸 The model learns a mapping from inputs (features) to known outputs (labels).
🔸 The goal is to **predict outputs** for unseen data.

---

## 🧩 Key Concepts

* **Features (X)**: Input variables used for prediction.
* **Labels (Y)**: Output or target variable to predict.
* **Model**: A mathematical representation learned from data.
* **Loss Function**: Measures how far predictions are from true values.
* **Optimizer**: Algorithm to minimize the loss (e.g., Gradient Descent).

---

## 📂 Types of Supervised Learning

### 🔹 Regression

* Output: Continuous values
* Example: Predicting house prices

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

### 🔹 Classification

* Output: Categories or classes
* Example: Spam detection, disease diagnosis

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
```

---

## ⚙️ Common Algorithms

| Algorithm                             | Type           | Use Case                |
| ------------------------------------- | -------------- | ----------------------- |
| Linear Regression                     | Regression     | Price prediction        |
| Logistic Regression                   | Classification | Binary classification   |
| Decision Tree                         | Both           | Interpretable models    |
| Random Forest                         | Both           | Ensemble learning       |
| K-Nearest Neighbors (KNN)             | Both           | Lazy learning           |
| Support Vector Machine (SVM)          | Both           | High-dimensional data   |
| Naive Bayes                           | Classification | Text classification     |
| Gradient Boosting (XGBoost, LightGBM) | Both           | High-performance models |

---

## 🧠 Important Terminologies

### 🎯 Target Variable

The actual value you're trying to predict.

### 📈 Overfitting

Model performs well on training data but poorly on test data. (Too complex)

### 📉 Underfitting

Model performs poorly on both training and test data. (Too simple)

### 🧪 Cross-Validation

Splitting data into folds to train/test multiple times for stable results.

### ⚖️ Bias-Variance Tradeoff

* **Bias**: Error due to overly simplistic model assumptions.
* **Variance**: Error due to model being too sensitive to training data.

### 🔄 Epoch

One complete pass through the training dataset (mainly in deep learning).

---

## 📊 Model Evaluation Metrics

### For Regression:

* **MAE**: Mean Absolute Error
* **MSE**: Mean Squared Error
* **RMSE**: Root Mean Squared Error
* **R² Score**: Proportion of variance explained

```python
from sklearn.metrics import mean_squared_error, r2_score
```

### For Classification:

* **Accuracy**: (Correct / Total)
* **Precision**: TP / (TP + FP)
* **Recall**: TP / (TP + FN)
* **F1 Score**: Harmonic mean of precision and recall
* **Confusion Matrix**: Shows TP, FP, FN, TN

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

---

## 🧪 Training vs Testing

| Term               | Purpose                                  |
| ------------------ | ---------------------------------------- |
| **Training Set**   | Used to fit the model                    |
| **Validation Set** | Used for tuning hyperparameters          |
| **Test Set**       | Used to evaluate final model performance |

---

## ✅ Tips & Best Practices

* Always **split** your data into training and testing sets.
* Use **cross-validation** to avoid overfitting.
* Try **scaling features** when using distance-based models (KNN, SVM).
* Perform **feature selection** to remove irrelevant columns.
* Use pipelines to ensure consistent data preprocessing and modeling.

---

## 📚 References

* [Scikit-learn Docs](https://scikit-learn.org/stable/)
* [ML Glossary by Google](https://developers.google.com/machine-learning/glossary)
* [Coursera ML Course by Andrew Ng](https://www.coursera.org/learn/machine-learning)

---
