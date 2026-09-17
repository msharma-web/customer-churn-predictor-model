# Telco Customer Churn Prediction

An end-to-end binary classification model built using Python, `pandas`, and `scikit-learn` to predict customer attrition based on demographic data, account attributes, and service usage patterns.

---

## **Project Overview**

Customer churn is a critical metric for subscription-based business models. This repository implements a machine learning workflow that cleans real-world telecom customer data, performs type conversions and categorical encoding, and trains a Random Forest Classifier to identify high-risk churn customers.

---

## **Key Features & Data Preprocessing**

* **Data Cleaning & Dimensionality Reduction:** Drops uninformative metadata (e.g., geographic coordinates, churn reasons) to prevent feature redundancy.
* **Optimized Memory Management:** Uses integer (`int8`, `int32`) and floating-point (`float32`) type casting to reduce memory overhead and optimize execution speed.
* **Feature Encoding:** Replaces binary survey responses (`Yes`/`No`) with numerical values (`1`/`0`) and applies `pd.get_dummies` for multi-class categorical features.
* **Stratified Classification:** Utilizes `scikit-learn`'s `RandomForestClassifier` with a stratified train-test split to preserve natural target distribution.

---

## **Repository Structure**

```text
├── TelcoCustomerChurn.csv   # Source dataset
├── churn_model.py          # Preprocessing script & model training pipeline
└── README.md               # Project documentation
