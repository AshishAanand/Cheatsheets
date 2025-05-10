# Scikit-learn Cheat Sheet

<div align="center">
  <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width="300" alt="Scikit-learn logo">
</div>

## Table of Contents

- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Data Preparation](#data-preparation)
  - [Loading Datasets](#loading-datasets)
  - [Train-Test Split](#train-test-split)
  - [Data Transformation](#data-transformation)
  - [Feature Selection](#feature-selection)
- [Supervised Learning](#supervised-learning)
  - [Linear Models](#linear-models)
  - [Support Vector Machines](#support-vector-machines)
  - [Decision Trees and Ensembles](#decision-trees-and-ensembles)
  - [Nearest Neighbors](#nearest-neighbors)
  - [Naive Bayes](#naive-bayes)
  - [Neural Networks](#neural-networks)
- [Unsupervised Learning](#unsupervised-learning)
  - [Clustering](#clustering)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Anomaly Detection](#anomaly-detection)
- [Model Selection and Evaluation](#model-selection-and-evaluation)
  - [Cross-Validation](#cross-validation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Metrics](#metrics)
  - [Pipeline](#pipeline)
- [Ensemble Methods](#ensemble-methods)
- [Model Persistence](#model-persistence)
- [Tips and Best Practices](#tips-and-best-practices)
- [Resources](#resources)

## Introduction

Scikit-learn (sklearn) is a Python library for machine learning built on NumPy, SciPy, and Matplotlib. It provides simple and efficient tools for data mining and data analysis, and is accessible to everybody.

## Installation and Setup

```python
# Installation
pip install scikit-learn

# Basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics, model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Setting random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

## Data Preparation

### Loading Datasets

```python
# Built-in datasets
from sklearn import datasets

# Small standard datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
diabetes = datasets.load_diabetes()
boston = datasets.load_boston()  # Note: Deprecated in newer versions

# Use as numpy arrays
X = iris.data  # Features
y = iris.target  # Target variable
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to pandas DataFrame
import pandas as pd
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Larger datasets
olivetti_faces = datasets.fetch_olivetti_faces()
20newsgroups = datasets.fetch_20newsgroups()

# Generate synthetic data
from sklearn.datasets import make_classification, make_regression, make_blobs, make_circles

# Classification data
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=2, n_redundant=10,
                          random_state=RANDOM_STATE)

# Regression data
X, y = make_regression(n_samples=1000, n_features=10, 
                      noise=0.1, random_state=RANDOM_STATE)

# Clustered data
X, y = make_blobs(n_samples=1000, centers=5, 
                 n_features=2, random_state=RANDOM_STATE)

# Non-linear data
X, y = make_circles(n_samples=1000, noise=0.1, 
                   factor=0.2, random_state=RANDOM_STATE)
```

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Simple train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Stratified split (maintains the same proportion of y classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Multi-split (train-validation-test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE
)  # 0.25 * 0.8 = 0.2
```

### Data Transformation

#### Scaling and Normalization

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

# Standardization (zero mean and unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler for test set

# Min-Max Scaling (features between [0,1])
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)

# Robust Scaling (uses median and quantiles, less sensitive to outliers)
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)

# Normalization (scale samples to unit norm)
normalizer = Normalizer()
X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)
```

#### Encoding Categorical Features

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Label Encoding (integers for categories, for target variable)
le = LabelEncoder()
y_encoded = le.fit_transform(y_categorical)
# Inverse transform to get original labels
y_original = le.inverse_transform(y_encoded)

# One-Hot Encoding (for nominal features)
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X_categorical)

# Ordinal Encoding (for ordered categories)
ord_encoder = OrdinalEncoder()
X_ordinal = ord_encoder.fit_transform(X_categorical)

# Using ColumnTransformer to apply different transformations
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [0, 1, 2]),  # Numerical columns indices
        ('cat', OneHotEncoder(), [3, 4])       # Categorical columns indices
    ])

X_processed = preprocessor.fit_transform(X)
```

#### Handling Missing Values

```python
from sklearn.impute import SimpleImputer

# Fill missing values with mean
imputer = SimpleImputer(strategy='mean')  # Other strategies: 'median', 'most_frequent', 'constant'
X_imputed = imputer.fit_transform(X_with_missing_values)

# For more advanced imputation
from sklearn.experimental import enable_iterative_imputer  # Enable experimental feature
from sklearn.impute import IterativeImputer

# Uses regression to estimate missing values
it_imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)
X_it_imputed = it_imputer.fit_transform(X_with_missing_values)

# KNN imputation
from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5)
X_knn_imputed = knn_imputer.fit_transform(X_with_missing_values)
```

### Feature Selection

```python
# Feature Selection using SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

# Select top K features based on ANOVA F-test
selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(X_train, y_train)

# Select features based on chi-squared test (for non-negative features)
selector = SelectKBest(chi2, k=5)
X_new = selector.fit_transform(X_train_minmax, y_train)  # Note: Features must be non-negative

# Select features based on mutual information
selector = SelectKBest(mutual_info_classif, k=5)
X_new = selector.fit_transform(X_train, y_train)

# Get selected feature indices
selected_indices = selector.get_support(indices=True)

# Get feature scores
feature_scores = selector.scores_

# Recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(max_iter=1000)
selector = RFE(estimator, n_features_to_select=5, step=1)
X_new = selector.fit_transform(X_train, y_train)

# Get feature ranking (lower is better)
feature_ranking = selector.ranking_

# Feature importance from models
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)
importances = model.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]
```

## Supervised Learning

### Linear Models

#### Linear Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Ordinary Least Squares
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Model coefficients and intercept
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_}")

# R² score
print(f"R² score: {lr.score(X_test, y_test)}")

# Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
ridge.fit(X_train, y_train)

# Lasso Regression (L1 regularization)
lasso = Lasso(alpha=0.1, random_state=RANDOM_STATE)
lasso.fit(X_train, y_train)

# ElasticNet (L1 + L2 regularization)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE)
elastic.fit(X_train, y_train)
```

#### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# Binary classification
logreg = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)  # Probability estimates

# Multi-class classification
logreg_multi = LogisticRegression(
    C=1.0, 
    penalty='l2',
    multi_class='multinomial',  # 'ovr' (one-vs-rest) or 'multinomial'
    solver='lbfgs',
    max_iter=1000,
    random_state=RANDOM_STATE
)
logreg_multi.fit(X_train, y_train)

# Model coefficients and intercept
print(f"Coefficients: {logreg.coef_}")
print(f"Intercept: {logreg.intercept_}")

# Evaluation
accuracy = logreg.score(X_test, y_test)
```

### Support Vector Machines

```python
from sklearn.svm import SVC, SVR, LinearSVC

# SVM for classification
svc = SVC(
    C=1.0,                 # Regularization parameter
    kernel='rbf',          # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',         # Kernel coefficient
    probability=True,      # Enable probability estimates
    random_state=RANDOM_STATE
)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
y_pred_proba = svc.predict_proba(X_test)  # If probability=True

# Linear SVM (faster for linear kernel)
lin_svc = LinearSVC(
    C=1.0,
    penalty='l2',
    loss='hinge',
    dual=True,
    random_state=RANDOM_STATE
)
lin_svc.fit(X_train, y_train)

# SVM for regression
svr = SVR(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    epsilon=0.1  # Epsilon in the epsilon-SVR model
)
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
```

### Decision Trees and Ensembles

#### Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# Decision tree for classification
dt_clf = DecisionTreeClassifier(
    criterion='gini',           # 'gini' or 'entropy'
    max_depth=5,                # Max tree depth
    min_samples_split=2,        # Min samples required to split
    min_samples_leaf=1,         # Min samples required at leaf node
    random_state=RANDOM_STATE
)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)

# Decision tree for regression
dt_reg = DecisionTreeRegressor(
    criterion='mse',            # 'mse', 'friedman_mse', 'mae'
    max_depth=5,
    random_state=RANDOM_STATE
)
dt_reg.fit(X_train, y_train)

# Feature importance
importance = dt_clf.feature_importances_

# Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(dt_clf, 
         filled=True, 
         feature_names=feature_names, 
         class_names=[str(c) for c in target_names],
         rounded=True)
plt.show()
```

#### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Random forest for classification
rf_clf = RandomForestClassifier(
    n_estimators=100,           # Number of trees
    criterion='gini',           # 'gini' or 'entropy'
    max_depth=None,             # No maximum depth
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',        # Number of features to consider: 'auto', 'sqrt', 'log2'
    bootstrap=True,             # Whether to use bootstrapped samples
    oob_score=True,             # Use out-of-bag samples for estimation
    n_jobs=-1,                  # Use all available cores
    random_state=RANDOM_STATE
)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
y_pred_proba = rf_clf.predict_proba(X_test)

# Random forest for regression
rf_reg = RandomForestRegressor(
    n_estimators=100,
    criterion='mse',            # 'mse', 'mae'
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf_reg.fit(X_train, y_train)

# Out-of-bag score
oob_score = rf_clf.oob_score_

# Feature importance
importance = rf_clf.feature_importances_
```

#### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Gradient boosting for classification
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,          # Shrinks contribution of each tree
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,              # Fraction of samples for fitting trees
    max_features=None,
    random_state=RANDOM_STATE
)
gb_clf.fit(X_train, y_train)
y_pred = gb_clf.predict(X_test)
y_pred_proba = gb_clf.predict_proba(X_test)

# Gradient boosting for regression
gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=RANDOM_STATE
)
gb_reg.fit(X_train, y_train)

# Feature importance
importance = gb_clf.feature_importances_
```

#### XGBoost (requires xgboost package)

```python
# Install: pip install xgboost
from xgboost import XGBClassifier, XGBRegressor

# XGBoost for classification
xgb_clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,       # Fraction of features for each tree
    objective='binary:logistic',
    n_jobs=-1,
    random_state=RANDOM_STATE
)
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
y_pred_proba = xgb_clf.predict_proba(X_test)

# XGBoost for regression
xgb_reg = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=RANDOM_STATE
)
xgb_reg.fit(X_train, y_train)

# Feature importance
importance = xgb_clf.feature_importances_
```

#### LightGBM (requires lightgbm package)

```python
# Install: pip install lightgbm
from lightgbm import LGBMClassifier, LGBMRegressor

# LightGBM for classification
lgb_clf = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
lgb_clf.fit(X_train, y_train)
y_pred = lgb_clf.predict(X_test)
y_pred_proba = lgb_clf.predict_proba(X_test)

# LightGBM for regression
lgb_reg = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
lgb_reg.fit(X_train, y_train)
```

### Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

# K-Nearest Neighbors for classification
knn_clf = KNeighborsClassifier(
    n_neighbors=5,             # Number of neighbors
    weights='uniform',         # 'uniform' or 'distance'
    algorithm='auto',          # 'auto', 'ball_tree', 'kd_tree', 'brute'
    p=2,                       # Power parameter for Minkowski metric (p=1: Manhattan, p=2: Euclidean)
    n_jobs=-1
)
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)
y_pred_proba = knn_clf.predict_proba(X_test)

# K-Nearest Neighbors for regression
knn_reg = KNeighborsRegressor(
    n_neighbors=5,
    weights='uniform',
    n_jobs=-1
)
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)

# Find nearest neighbors
nn = NearestNeighbors(
    n_neighbors=5,
    algorithm='auto',
    n_jobs=-1
)
nn.fit(X)
distances, indices = nn.kneighbors(X_query)
```

### Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB

# Gaussian Naive Bayes (for continuous data)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_pred_proba = gnb.predict_proba(X_test)

# Multinomial Naive Bayes (for discrete counts, e.g., text)
mnb = MultinomialNB(alpha=1.0)  # Laplace smoothing parameter
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

# Bernoulli Naive Bayes (for binary/boolean features)
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)

# Complement Naive Bayes (good for imbalanced datasets)
cnb = ComplementNB(alpha=1.0)
cnb.fit(X_train, y_train)
y_pred = cnb.predict(X_test)
```

### Neural Networks

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Multi-layer Perceptron for classification
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100,),      # Tuple with number of neurons per hidden layer
    activation='relu',              # 'identity', 'logistic', 'tanh', 'relu'
    solver='adam',                  # 'lbfgs', 'sgd', 'adam'
    alpha=0.0001,                   # L2 penalty parameter
    batch_size='auto',              # Size of mini-batches for stochastic optimizers
    learning_rate='constant',       # 'constant', 'invscaling', 'adaptive'
    learning_rate_init=0.001,       # Initial learning rate
    max_iter=200,                   # Maximum number of iterations
    early_stopping=False,           # Whether to use early stopping
    validation_fraction=0.1,        # Fraction of training data for validation
    random_state=RANDOM_STATE
)
mlp_clf.fit(X_train, y_train)
y_pred = mlp_clf.predict(X_test)
y_pred_proba = mlp_clf.predict_proba(X_test)

# Multi-layer Perceptron for regression
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    max_iter=200,
    random_state=RANDOM_STATE
)
mlp_reg.fit(X_train, y_train)
y_pred = mlp_reg.predict(X_test)

# Access network attributes
print(f"Loss curve: {mlp_clf.loss_curve_}")  # Loss at each iteration
print(f"Number of iterations: {mlp_clf.n_iter_}")
print(f"Number of layers: {mlp_clf.n_layers_}")
```

## Unsupervised Learning

### Clustering

#### K-Means

```python
from sklearn.cluster import KMeans

# K-Means clustering
kmeans = KMeans(
    n_clusters=3,               # Number of clusters
    init='k-means++',           # Initialization method: 'k-means++', 'random'
    n_init=10,                  # Number of initializations
    max_iter=300,               # Maximum number of iterations
    tol=1e-4,                   # Tolerance for convergence
    random_state=RANDOM_STATE
)
kmeans.fit(X)

# Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_
inertia = kmeans.inertia_  # Sum of squared distances to closest centroid

# Predict clusters for new data
cluster_labels = kmeans.predict(X_new)

# Finding optimal number of clusters using Elbow Method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

#### DBSCAN

```python
from sklearn.cluster import DBSCAN

# DBSCAN clustering
dbscan = DBSCAN(
    eps=0.5,                   # Maximum distance between two samples
    min_samples=5,             # Number of samples in a neighborhood for a point to be a core point
    metric='euclidean',        # Distance metric
    n_jobs=-1
)
clusters = dbscan.fit_predict(X)

# Cluster labels
labels = dbscan.labels_  # -1 represents noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Number of clusters (excluding noise)
```

#### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Hierarchical clustering
agg_clustering = AgglomerativeClustering(
    n_clusters=3,                  # Number of clusters
    affinity='euclidean',          # Distance metric: 'euclidean', 'l1', 'l2', 'manhattan', 'cosine'
    linkage='ward'                 # Linkage criterion: 'ward', 'complete', 'average', 'single'
)
clusters = agg_clustering.fit_predict(X)

# Visualize dendrogram (using SciPy)
Z = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
```

#### Gaussian Mixture Models

```python
from sklearn.mixture import GaussianMixture

# Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=3,            # Number of mixture components
    covariance_type='full',    # 'full', 'tied', 'diag', 'spherical'
    max_iter=100,              # Maximum number of iterations
    random_state=RANDOM_STATE
)
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Get probability of each sample belonging to each cluster
proba = gmm.predict_proba(X)

# Get Bayesian Information Criterion
bic = gmm.bic(X)

# Get Akaike Information Criterion
aic = gmm.aic(X)

# Finding optimal number of components using BIC
bics = []
for n in range(1, 11):
    gmm = GaussianMixture(n_components=n, random_state=RANDOM_STATE)
    gmm.fit(X)
    bics.append(gmm.bic(X))

plt.plot(range(1, 11), bics, marker='o')
plt.xlabel('Number of components')
plt.ylabel('BIC')
plt.title('BIC for different numbers of components')
plt.show()
```

### Dimensionality Reduction

#### Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA

# PCA
pca = PCA(
    n_components=2,            # Number of components (or fraction of variance to preserve)
    whiten=False,              # Whitening (divide by singular values to ensure uncorrelated outputs)
    random_state=RANDOM_STATE
)
X_pca = pca.fit_transform(X)

# Get explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Get components (loadings)
components = pca.components_

# Visualize explained variance
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', color='red')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Component')
plt.show()

# Visualize 2D projection
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Projection')
plt.colorbar(label='Class')
plt.show()
```

#### t-SNE

```python
from sklearn.manifold import TSNE

# t-SNE
tsne = TSNE(
    n_components=2,            # Number of components
    perplexity=30.0,           # Related to the number of nearest neighbors
    early_exaggeration=12.0,   # Controls how tight clusters are
    learning_rate=200.0,       # Learning rate
    n_iter=1000,               # Number of iterations
    metric='euclidean',        # Distance metric
    init='pca',                # Initialization method: 'pca', 'random'
    random_state=RANDOM_STATE
)
X_tsne = tsne.fit_transform(X)

# Visualize t-SNE projection
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Projection')
plt.colorbar(label='Class')
plt.show()
```

#### UMAP (requires umap-