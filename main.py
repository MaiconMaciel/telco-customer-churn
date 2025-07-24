import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from imblearn.over_sampling import SMOTE

# Load dataset
file_path = 'dataset/Telco-customer-churn.csv'
df = pd.read_csv(file_path)

# Basic preprocessing
df = df.drop(columns=['customerID'])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Encode binary categorical features
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encode categorical features
categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
numeric_cols = ['MonthlyCharges', 'TotalCharges']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Train logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and Youden's J statistic
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
youden_index = tpr - fpr
best_threshold_index = np.argmax(youden_index)
best_threshold = thresholds[best_threshold_index]

print(f"Optimal threshold (Youden's J): {best_threshold:.2f}")

# Predict using optimal threshold
y_pred_opt = (y_proba >= best_threshold).astype(int)

# Classification report
print("\nClassification Report with Optimized Threshold:")
print(classification_report(y_test, y_pred_opt))

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix (Threshold = {best_threshold:.2f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC curve with optimal threshold point
plt.plot(fpr, tpr, label='ROC Curve')
plt.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='red', label='Optimal Threshold')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Optimal Threshold')
plt.legend()
plt.grid()
plt.show()
