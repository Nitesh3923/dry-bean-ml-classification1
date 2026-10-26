# Dry Bean Classification - Machine Learning Assignment 2

## A. Problem Statement

The objective of this project is to implement, evaluate, and compare
five machine learning classification models for predicting dry bean
classes using numerical shape and dimensional characteristics.

The trained models are also integrated into an interactive Streamlit
application. The application evaluates previously trained models on
uploaded test data without retraining the models.

---

## B. Dataset Description

**Dataset:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository

**Problem Type:** Multiclass Classification

**Target Variable:** Class

### Dataset Details

- Original instances: 13,611
- Instances after duplicate removal: 13,543
- Input features: 16 numerical features
- Target classes: 7
- Missing values: 0
- Training/Test split: 80:20
- Random state: 42
- Stratified splitting: Yes

### Target Classes

The dataset contains the following seven bean classes:

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

---

## C. GitHub Repository Link

GitHub Repository:

**To be added after GitHub repository creation**

---

## D. Models Used

The following five classification models were implemented on the
same Dry Bean dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine

### Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

For multiclass evaluation, macro averaging was used for Precision,
Recall, F1 Score and One-vs-Rest AUC.

---

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9947 | 0.9307 | 0.9300 | 0.9302 | 0.9023 |
| Decision Tree | 0.9029 | 0.9649 | 0.9171 | 0.9168 | 0.9170 | 0.8826 |
| KNN | 0.9136 | 0.9854 | 0.9268 | 0.9237 | 0.9250 | 0.8956 |
| Naive Bayes | 0.7630 | 0.9670 | 0.7736 | 0.7685 | 0.7669 | 0.7143 |
| Support Vector Machine | 0.9243 | 0.9945 | 0.9372 | 0.9355 | 0.9363 | 0.9085 |

---

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression achieved strong and balanced classification performance. It obtained the highest AUC score of 0.9947 and achieved the second-highest F1 and MCC scores. |
| Decision Tree | Decision Tree achieved good classification performance, but its Accuracy, F1 and MCC were lower than Logistic Regression, KNN and Support Vector Machine. |
| KNN | KNN demonstrated strong performance with an Accuracy of 0.9136, F1 Score of 0.9250 and MCC of 0.8956. Its overall performance was slightly below Logistic Regression and Support Vector Machine. |
| Naive Bayes | Naive Bayes produced the lowest Accuracy, Precision, Recall, F1 and MCC among the five evaluated classifiers. |
| Support Vector Machine | Support Vector Machine achieved the highest Accuracy, Precision, Recall, F1 and MCC among all five classifiers and demonstrated the strongest overall classification performance. |

---

## Overall Winner

**Support Vector Machine (SVM)** is the overall winning model for
this dataset.

Its evaluation results are:

- Accuracy: 0.9243
- AUC: 0.9945
- Precision: 0.9372
- Recall: 0.9355
- F1 Score: 0.9363
- MCC: 0.9085

SVM achieved the highest Accuracy, Precision, Recall, F1 Score and
MCC among the five evaluated models.

Logistic Regression achieved a slightly higher AUC score of 0.9947
compared with the SVM AUC of 0.9945. However, SVM performed better
on the other major evaluation metrics and was therefore selected as
the overall winner.

---

## Streamlit Application

Live Streamlit Application:

**https://dry-bean-ml-classification-n2tus4kckpw9d27zxm9vra.streamlit.app/**

The Streamlit application provides:

- CSV test dataset upload
- Classification model-selection dropdown
- Accuracy display
- AUC Score display
- Precision display
- Recall display
- F1 Score display
- MCC Score display
- Confusion matrix
- Classification report
- Prediction results

The application loads previously trained model pipelines and
evaluates the uploaded test dataset without retraining the models.

---

## Project Structure

ML_Assignment_2/

- app.py
- requirements.txt
- README.md
- test_data.csv
- model_metrics.csv
- ML_Assignment_2.ipynb
- model/
  - logistic_regression.joblib
  - decision_tree.joblib
  - knn.joblib
  - naive_bayes.joblib
  - support_vector_machine.joblib
  - label_encoder.joblib
  - feature_names.joblib
  - project_metadata.joblib
- reports/
  - classification reports
  - confusion matrices

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn

---

## Author

Nitesh Kumar

Machine Learning Assignment 2
M.Tech (AIML)
Work Integrated Learning Programmes
BITS Pilani
