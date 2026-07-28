# 🏥 Hospital Call Center Routing using MLflow
<img width="1907" height="882" alt="image" src="https://github.com/user-attachments/assets/f911cbe9-d6bc-4e95-9c83-b865e3d9f835" />


## 📌 Project Overview

This project predicts the **Recommended Hospital Department** for incoming patient calls using Machine Learning.

The project demonstrates a complete ML workflow using **MLflow**, including:

- Data preprocessing
- Model training
- Model evaluation
- Experiment tracking
- Model logging
- Model registration

---

## 🎯 Objective

To classify patient calls into the correct hospital department based on call details and patient information, reducing manual routing and improving response time.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Matplotlib

---

## 📂 Project Structure

```
Hospital-callcenter-mlflow
│
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Workflow

### 1. Data Preprocessing

- Load dataset
- Remove unnecessary columns
- Encode categorical features
- Split data into training and testing sets

---

### 2. Model Training

The following models were trained:

- Logistic Regression
- Decision Tree
- Random Forest

For every model, MLflow logs:

- Parameters
- Metrics
- Artifacts
- Model

---

### 3. Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score

A Classification Report is also generated.

---

## 📊 MLflow Experiment Tracking

MLflow was used to compare multiple machine learning models.

### Logged Models

- Logistic Regression
- Decision Tree
- Random Forest

---

## 📈 Example Metrics

| Metric | Decision Tree |
|---------|--------------|
| Accuracy | 0.794 |
| Precision | 0.794 |
| Recall | 0.794 |
| F1 Score | 0.793 |

---

# 📸 MLflow Screenshots

## Experiment Metrics

![MLflow Metrics](images/mlflow_metrics.png)

---

## Registered Models

![MLflow Models](images/mlflow_models.png)

---

## ▶️ How to Run

Install dependencies

```bash
pip install -r requirements.txt
```

Train models

```bash
python src/train.py
```

Evaluate model

```bash
python src/evaluate.py
```

Run prediction

```bash
python src/predict.py
```

Launch MLflow UI

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## 📌 Features

✅ Data preprocessing

✅ Multiple ML models

✅ MLflow experiment tracking

✅ Model comparison

✅ Registered models

✅ Performance metrics

✅ Classification report generation

---

## 👩‍💻 Author

**Akshata Rajesh Ram**

M.Sc. Data Science & Artificial Intelligence

SRH University, Germany
