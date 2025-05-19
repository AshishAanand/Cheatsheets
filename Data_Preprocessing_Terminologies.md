# 📊 Data Preprocessing Terminologies Cheat Sheet

> A quick reference guide to essential **data preprocessing** concepts used in Machine Learning & Data Science.
> Use this to refresh your knowledge before modeling or while working with any dataset!

---

## 📌 Table of Contents

* [What is Data Preprocessing?](#what-is-data-preprocessing)
* [1. Data Cleaning](#1-data-cleaning)

  * Missing Values
  * Outliers
  * Duplicates
* [2. Data Transformation](#2-data-transformation)

  * Normalization vs Standardization
  * Encoding Categorical Data
  * Binning
* [3. Feature Engineering](#3-feature-engineering)

  * Feature Extraction
  * Feature Selection
* [4. Data Integration](#4-data-integration)
* [5. Data Reduction](#5-data-reduction)
* [Tips & Best Practices](#tips--best-practices)

---

## What is Data Preprocessing?

Data preprocessing is the **first step** in a machine learning workflow. It involves transforming raw data into a clean, usable format.

### Goals:

* Improve model accuracy
* Handle inconsistencies and missing values
* Make data suitable for algorithms

---

## 1. 🧹 Data Cleaning

### 🔹 Missing Values

Techniques to handle null or NaN values:

```python
# Drop rows with missing values
df.dropna()

# Fill missing with a value
df.fillna(0)

# Fill using mean/median/mode
df['col'].fillna(df['col'].mean(), inplace=True)
```

### 🔹 Outliers

Values that deviate significantly from others in the dataset.

```python
# Using IQR to detect outliers
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['col'] < Q1 - 1.5*IQR) | (df['col'] > Q3 + 1.5*IQR)]
```

### 🔹 Duplicates

```python
# Drop duplicate rows
df.drop_duplicates(inplace=True)
```

---

## 2. 🔁 Data Transformation

### 🔹 Normalization (Min-Max Scaling)

Scales values between 0 and 1.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['col']])
```

### 🔹 Standardization (Z-score Scaling)

Centers the data (mean = 0, std = 1).

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized = scaler.fit_transform(df[['col']])
```

### 🔹 Encoding Categorical Data

* **Label Encoding**

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
```

* **One-Hot Encoding**

```python
pd.get_dummies(df, columns=['category'])
```

### 🔹 Binning

Convert continuous data into categorical bins.

```python
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60], labels=["Child", "Teen", "Adult", "Senior"])
```

---

## 3. 🧠 Feature Engineering

### 🔹 Feature Extraction

Creating new features from existing data (e.g., extracting year from a date).

```python
df['year'] = pd.to_datetime(df['date']).dt.year
```

### 🔹 Feature Selection

Removing irrelevant or redundant features.

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)
```

---

## 4. 🔗 Data Integration

Combining data from multiple sources into a single dataset.

```python
# Merge two datasets on a key
merged_df = pd.merge(df1, df2, on='id', how='inner')
```

---

## 5. 📉 Data Reduction

Reducing dataset size without losing information.

* **Principal Component Analysis (PCA)**:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
```

* **Sampling**: Select a subset of data for faster processing.

---

## 💡 Tips & Best Practices

* Always **visualize missing values** and outliers before acting.
* **Normalize** when using distance-based models (e.g., KNN, SVM).
* **Standardize** when using linear models (e.g., Logistic Regression).
* Keep track of your transformations (e.g., save scalers/encoders).
* Use pipelines for consistency across training and testing.

---

## 📚 References

* [Scikit-learn Preprocessing Docs](https://scikit-learn.org/stable/modules/preprocessing.html)
* [Pandas Docs](https://pandas.pydata.org/docs/)
* [Awesome Data Science](https://github.com/academic/awesome-datascience)

---
