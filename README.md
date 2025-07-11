# Telco Customer Churn Prediction

This project predicts whether a telecom customer will churn using machine learning.

## 📌 Dataset & Problem
- Dataset: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Problem: Binary classification to predict churn (`Yes` / `No`).

---

## 🛠️ Preprocessing
- Handled missing values with KNNImputer.
- Label Encoding for binary features.
- One-Hot Encoding for multiclass features.
- StandardScaler for numeric columns.
- Outlier clipping at 1st and 99th percentiles.
- Handled class imbalance with SMOTE.

---

## 🤖 Models & Training
- Logistic Regression
- Random Forest
- XGBoost
- AdaBoost
- Voting Classifier
- Random Forest with GridSearchCV for hyperparameter tuning.

---

## ✅ Results
| Model              | Accuracy | F1-score | ROC AUC |
|---------------------|---------|----------|---------|
| Best Random Forest  | 0.76    | 0.60     | ~0.79   |

- Best Random Forest parameters found with GridSearchCV.
- Visualized Confusion Matrix and ROC Curve.

---

## 📈 Deployment
- Saved best model with `joblib`.
- Built a **Streamlit app** where users can input customer data and see churn prediction.
- Included probability-based visualization.

---

