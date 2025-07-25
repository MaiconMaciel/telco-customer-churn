# Telco Customer Churn Prediction

## - Project Overview

This project aims to build a machine learning model to **predict customer churn** for a fictional telecommunications company. **Churn:** when customers stop using a service, is a key business problem that directly impacts revenue. Identifying customers at risk of churning allows companies to take proactive retention actions.

This was developed as a **study project**, exploring core steps of a data science workflow, including:

- Exploratory Data Analysis (EDA)
- Data preprocessing and encoding
- Handling class imbalance using **SMOTE**
- Training a **Logistic Regression** classifier
- Finding the **optimal decision threshold** using **Youden's J statistic**
- Evaluating model performance using precision, recall, F1-score, and ROC-AUC

The focus is on **interpretability**, **best practices**, and understanding **real-world modeling challenges**, such as imbalanced datasets and threshold tuning.

---

## 📂 Dataset

The dataset used is the **Telco Customer Churn** dataset, which is publicly available on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). It contains information about:

- Demographics (gender, age, etc.)
- Service subscriptions (internet, phone, streaming)
- Billing and contract details
- Churn label (Yes/No)

You can download it directly from Kaggle or use the file provided in the `/data` folder (i do not own any rights to the data. All rights to the Original Authors).

---

## 👩‍💻 Technologies Used

- Python 3
- Pandas & NumPy
- Matplotlib & Seaborn (visualizations)
- Scikit-learn (modeling & evaluation)
- Imbalanced-learn (SMOTE)

---

## 📌 Goal

Build a clear, interpretable, and statistically sound **classification model** to predict customer churn, while showcasing:
- Data cleaning and preprocessing techniques
- Handling class imbalance
- Model evaluation and performance optimization

---

> 💡 This is intended for educational purposes and portfolio demonstration. Feel free to fork, clone, and adapt it!

---

## 📌 Features Used

- **Numerical**: MonthlyCharges, TotalCharges
- **Categorical**: Contract type, InternetService, PaymentMethod, etc.
- **Binary**: Partner, Dependents, PhoneService, PaperlessBilling
- One-hot encoding is applied to multi-class categorical variables.

---

## 🛠️ Technologies & Libraries

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (for SMOTE)

---

### 📈 Results

    Balanced accuracy and F1-score improved after using SMOTE

    Optimal threshold selected using ROC curve improves sensitivity vs default 0.5

    Confusion matrix and ROC plots help visualize performance

---

### 🧠 Key Learnings

    How to deal with imbalanced classification problems

    The importance of data cleaning and encoding

    Why threshold tuning can significantly impact binary classifier performance

    Using SMOTE to synthetically balance datasets

    How logistic regression can provide interpretable, probabilistic outputs

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/telco-churn-ml.git
   cd telco-churn-ml
   
2. (Optional) Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Launch Jupyter Notebook and open Telco-Churn-Prediction.ipynb:
    ```bash
    jupyter notebook
    ```
   
---

# 🙋‍♂️ Author

Maicon Costa Maciel

[LinkedIn](https://linkedin.com/in/maiconmaciel) • [GitHub](https://github.com/MaiconMaciel) • [CV](https://drive.google.com/drive/folders/1OJNzsRnyEsfJtAaD5R4xtCi6bU53Wacx?usp=sharing)