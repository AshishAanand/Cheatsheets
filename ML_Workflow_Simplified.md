# ML Workflow Simplified Cheat Sheet

A concise, developer-friendly overview of the end-to-end machine learning (ML) workflow. Use this as a quick reference for ML projects, from problem definition to deployment and monitoring.

---

## **1. Problem Definition**

- **Clarify the Objective:** Understand the business context and define what problem you are solving.
- **Set Success Metrics:** Identify KPIs (accuracy, F1-score, RMSE, etc.) relevant to your use case.
- **Stakeholder Alignment:** Ensure all stakeholders agree on goals and deliverables[1][5].

---

## **2. Data Collection & Preparation**

- **Data Sources:** Identify and acquire data from databases, APIs, logs, or external datasets[1][5].
- **Data Cleaning:** Handle missing values, outliers, duplicates, and errors.
- **Feature Engineering:** Create, transform, and select features to improve model performance.
- **Data Transformation:** Normalize, scale, encode categorical variables, and augment data if needed[6].
- **Data Splitting:**
  - **Training Set:** Used to train the model (typically 70-80%).
  - **Validation Set:** Used for hyperparameter tuning and model selection.
  - **Test Set:** Used for final evaluation (typically 20-30%)[1][6].

**Example (Python, using scikit-learn):**
```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

---

## **3. Exploratory Data Analysis (EDA)**

- **Visualize Data:** Use histograms, scatter plots, box plots to understand distributions and relationships.
- **Statistical Summary:** Check means, medians, correlations, and standard deviations.
- **Detect Patterns:** Identify trends, anomalies, and data leakage risks[1][7].

**Example (Python):**
```python
import seaborn as sns
sns.pairplot(df, hue='target')
```

---

## **4. Model Selection & Training**

- **Choose Algorithms:** Select based on problem type (classification, regression, clustering) and data characteristics.
- **Baseline Model:** Start with simple models to set a benchmark.
- **Train Models:** Fit algorithms on training data.
- **Cross-Validation:** Use k-fold or stratified sampling for robust evaluation[1][5][6].

**Example (Python):**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

## **5. Model Evaluation & Tuning**

- **Performance Metrics:** Use accuracy, precision, recall, F1-score, ROC-AUC (classification) or RMSE, MAE (regression).
- **Hyperparameter Tuning:** Grid search, random search, or Bayesian optimization.
- **Avoid Overfitting:** Regularization, dropout, early stopping as needed.
- **Iterate:** Refine features, try different models, and re-evaluate[1][6].

**Example (Grid Search with scikit-learn):**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
```

---

## **6. Model Deployment**

- **Model Serialization:** Save the trained model (e.g., with `pickle` or `joblib`).
- **API Serving:** Deploy as a REST API using Flask, FastAPI, or cloud services.
- **Batch vs. Real-Time:** Choose deployment strategy based on use case[4][7].

**Example (Save Model):**
```python
import joblib
joblib.dump(model, 'model.pkl')
```

---

## **7. Monitoring & Maintenance**

- **Monitor Performance:** Track model accuracy, latency, and drift in production.
- **Retrain as Needed:** Update the model with new data to maintain performance.
- **Version Control:** Manage model versions and document changes[4][3].

---

## **8. Common Tips & Best Practices**

- **Document Everything:** Data sources, preprocessing steps, model configs, and evaluation results.
- **Automate Pipelines:** Use tools like MLflow, Kubeflow, or Airflow for reproducible workflows.
- **Collaborate:** Work closely with data engineers, domain experts, and stakeholders for feedback and improvement[1][7].
- **Infrastructure:** Ensure your hardware and storage are ML-ready to prevent bottlenecks[1].

---

## **ML Workflow Overview Table**

| Step                    | Key Actions                                  | Tools/Libraries              |
|-------------------------|----------------------------------------------|------------------------------|
| Problem Definition      | Define goals, metrics                        | -                            |
| Data Collection/Prep    | Acquire, clean, engineer, split data         | pandas, numpy                |
| EDA                     | Visualize, summarize, detect issues          | matplotlib, seaborn          |
| Model Selection/Training| Choose/train models, cross-validation        | scikit-learn, TensorFlow     |
| Evaluation/Tuning       | Metrics, hyperparameter search, iteration    | scikit-learn, Optuna         |
| Deployment              | Serialize, serve, monitor                    | Flask, FastAPI, Docker       |
| Monitoring/Maintenance  | Track, retrain, version control              | MLflow, Prometheus           |

---

## **References for Further Reading**

- Pure Storage: [What Is a Machine Learning Workflow?][1]
- DataCamp: [A Beginner's Guide to The Machine Learning Workflow][5]
- Google Cloud: [Machine learning workflow | AI Platform][4]
- NVIDIA: [Machine Learning in Practice: ML Workflows][7]

---

**Keep this cheat sheet handy for your next ML project!**

Citations:
[1] https://www.purestorage.com/knowledge/machine-learning-workflow.html
[2] https://ml-ops.org/content/end-to-end-ml-workflow
[3] https://www.turing.com/kb/understanding-the-workflow-of-mlops
[4] https://cloud.google.com/ai-platform/docs/ml-solutions-overview
[5] https://www.datacamp.com/blog/a-beginner-s-guide-to-the-machine-learning-workflow
[6] https://sigma.ai/machine-learning-workflow/
[7] https://developer.nvidia.com/blog/machine-learning-in-practice-ml-workflows/
[8] https://www.gatevidyalay.com/machine-learning-workflow-process-steps/
[9] https://fastdatascience.com/data-science-project-management/workflows-pipelines-ml-ai/
[10] https://360digitmg.com/ml-workflow
[11] https://viso.ai/computer-vision/typical-workflow-for-building-a-machine-learning-model/
[12] https://www.codingdojo.com/blog/machine-learning-workflow
[13] https://www.run.ai/guides/machine-learning-engineering/machine-learning-workflow
[14] https://www.datagrads.com/the-machine-learning-workflow-explained/
[15] https://www.youtube.com/watch?v=Xy7eV8wRLbE

---
