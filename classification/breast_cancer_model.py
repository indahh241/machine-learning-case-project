# ==========================================
# 1. IMPORT LIBRARY
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 2. LOAD DATASET
# ==========================================
data = pd.read_csv("datasets/breast_cancer.csv")

print("Preview Data:")
print(data.head())

print("\nInfo Dataset:")
print(data.info())

print("\nStatistik:")
print(data.describe())

# ==========================================
# 3. DATA CLEANING
# ==========================================
data = data.drop(columns=[col for col in ["id", "Unnamed: 32"] if col in data.columns])

# ==========================================
# 4. ENCODING
# ==========================================
le = LabelEncoder()
data["diagnosis"] = le.fit_transform(data["diagnosis"])

# ==========================================
# 5. EDA
# ==========================================
sns.countplot(x="diagnosis", data=data)
plt.title("Distribusi Diagnosis")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.show()

# ==========================================
# 6. FEATURE & TARGET
# ==========================================
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================================
# 7. SPLITTING
# ==========================================
splits = {"70:30":0.3,"80:20":0.2,"90:10":0.1}
results = []

for name,test_size in splits.items():

    print(f"\n===== SPLIT {name} =====")

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=test_size,random_state=42,stratify=y
    )

    # Logistic Regression (Tuning)
    param_lr = {"C":[0.1,1,10]}
    grid_lr = GridSearchCV(LogisticRegression(max_iter=5000), param_lr, cv=5)
    grid_lr.fit(X_train,y_train)
    pred_lr = grid_lr.predict(X_test)
    acc_lr = accuracy_score(y_test,pred_lr)

    # SVM (Tuning)
    param_svm = {"C":[0.1,1,10],"kernel":["linear","rbf"]}
    grid_svm = GridSearchCV(SVC(), param_svm, cv=5)
    grid_svm.fit(X_train,y_train)
    pred_svm = grid_svm.predict(X_test)
    acc_svm = accuracy_score(y_test,pred_svm)

    # Random Forest (Tuning)
    param_rf = {
        "n_estimators":[100,200],
        "max_depth":[None,10]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_rf, cv=5)
    grid_rf.fit(X_train,y_train)
    pred_rf = grid_rf.predict(X_test)
    acc_rf = accuracy_score(y_test,pred_rf)

    print("LR:",acc_lr)
    print("SVM:",acc_svm)
    print("RF:",acc_rf)

    print("\nConfusion Matrix:")
    sns.heatmap(confusion_matrix(y_test,pred_rf),annot=True,fmt="d")
    plt.show()

    print(classification_report(y_test,pred_rf))

    results.append({
        "Split":name,
        "LR":acc_lr,
        "SVM":acc_svm,
        "RF":acc_rf
    })

# ==========================================
# 8. COMPARISON
# ==========================================
df = pd.DataFrame(results)
print(df)

df.set_index("Split").plot(kind="bar")
plt.title("Perbandingan Model")
plt.show()