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

print("=== HEAD ===")
print(data.head())

print("\n=== INFO ===")
print(data.info())

print("\n=== DESCRIPTIVE STATISTICS ===")
print(data.describe())

# ==========================================
# 3. DATA UNDERSTANDING
# ==========================================
print("\n=== MISSING VALUE ===")
print(data.isnull().sum())

# ==========================================
# 4. DATA CLEANING
# ==========================================
data = data.drop(columns=[col for col in ["id", "Unnamed: 32"] if col in data.columns])

# ==========================================
# 5. ENCODING
# ==========================================
le = LabelEncoder()
data["diagnosis"] = le.fit_transform(data["diagnosis"])

# ==========================================
# 6. EDA (EXPLORATORY DATA ANALYSIS)
# ==========================================

# Distribusi target
sns.countplot(x="diagnosis", data=data)
plt.title("Distribusi Diagnosis (0 = Malignant, 1 = Benign)")
plt.show()

# Heatmap korelasi
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()

# Boxplot (deteksi outlier)
plt.figure(figsize=(15,6))
sns.boxplot(data=data.drop("diagnosis", axis=1))
plt.xticks(rotation=90)
plt.title("Boxplot Deteksi Outlier")
plt.show()

# ==========================================
# 7. FEATURE & TARGET
# ==========================================
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# Normalisasi
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================================
# 8. SPLITTING
# ==========================================
splits = {"70:30":0.3,"80:20":0.2,"90:10":0.1}
results = []

# ==========================================
# 9. MODELING + EVALUATION
# ==========================================
for name, test_size in splits.items():

    print(f"\n========== SPLIT {name} ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ======================
    # LOGISTIC REGRESSION
    # ======================
    param_lr = {"C":[0.1,1,10]}
    grid_lr = GridSearchCV(LogisticRegression(max_iter=5000), param_lr, cv=5)
    grid_lr.fit(X_train,y_train)
    pred_lr = grid_lr.predict(X_test)
    acc_lr = accuracy_score(y_test,pred_lr)

    print("\n--- Logistic Regression ---")
    print("Accuracy:", acc_lr)
    print("Best Param:", grid_lr.best_params_)
    sns.heatmap(confusion_matrix(y_test,pred_lr),annot=True,fmt="d")
    plt.title("Confusion Matrix - LR")
    plt.show()
    print(classification_report(y_test,pred_lr))

    # ======================
    # SVM
    # ======================
    param_svm = {"C":[0.1,1,10],"kernel":["linear","rbf"]}
    grid_svm = GridSearchCV(SVC(), param_svm, cv=5)
    grid_svm.fit(X_train,y_train)
    pred_svm = grid_svm.predict(X_test)
    acc_svm = accuracy_score(y_test,pred_svm)

    print("\n--- SVM ---")
    print("Accuracy:", acc_svm)
    print("Best Param:", grid_svm.best_params_)
    sns.heatmap(confusion_matrix(y_test,pred_svm),annot=True,fmt="d")
    plt.title("Confusion Matrix - SVM")
    plt.show()
    print(classification_report(y_test,pred_svm))

    # ======================
    # RANDOM FOREST
    # ======================
    param_rf = {
        "n_estimators":[100,200],
        "max_depth":[None,10]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_rf, cv=5)
    grid_rf.fit(X_train,y_train)
    pred_rf = grid_rf.predict(X_test)
    acc_rf = accuracy_score(y_test,pred_rf)

    print("\n--- Random Forest ---")
    print("Accuracy:", acc_rf)
    print("Best Param:", grid_rf.best_params_)
    sns.heatmap(confusion_matrix(y_test,pred_rf),annot=True,fmt="d")
    plt.title("Confusion Matrix - RF")
    plt.show()
    print(classification_report(y_test,pred_rf))

    # Simpan hasil
    results.append({
        "Split":name,
        "LR":acc_lr,
        "SVM":acc_svm,
        "RF":acc_rf
    })

# ==========================================
# 10. MODEL COMPARISON
# ==========================================
df = pd.DataFrame(results)

print("\n=== HASIL PERBANDINGAN MODEL ===")
print(df)

df.set_index("Split").plot(kind="bar")
plt.title("Perbandingan Akurasi Model")
plt.ylabel("Accuracy")
plt.show()