# ==============================================
# IMPORT
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================
# LOAD DATA
# ==============================================
data = pd.read_csv("datasets/energy_efficiency.csv")

print("=== HEAD ===")
print(data.head())

print("\n=== INFO ===")
print(data.info())

print("\n=== DESCRIPTIVE STATISTICS ===")
print(data.describe())

# ==============================================
# DATA UNDERSTANDING
# ==============================================
print("\nJumlah missing value:\n", data.isnull().sum())

# ==============================================
# EDA (EXPLORATORY DATA ANALYSIS)
# ==============================================

# 1. DISTRIBUSI DATA
data.hist(figsize=(12,10))
plt.suptitle("Distribusi Data")
plt.show()

# 2. HEATMAP KORELASI
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), cmap="coolwarm", annot=True)
plt.title("Heatmap Korelasi")
plt.show()

# 3. BOXPLOT (DETEKSI OUTLIER)
plt.figure(figsize=(12,6))
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.title("Boxplot untuk Deteksi Outlier")
plt.show()

# ==============================================
# PREPROCESSING
# ==============================================

# (Jika ada missing value → handle)
# data = data.dropna()

# FEATURE & TARGET
X = data.drop("Heating_Load", axis=1)
y = data["Heating_Load"]

# NORMALISASI
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==============================================
# SPLITTING
# ==============================================
splits = {"70:30":0.3,"80:20":0.2,"90:10":0.1}
results = []

# ==============================================
# MODELING
# ==============================================
for name, test_size in splits.items():

    print(f"\n================ {name} =================")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # =========================
    # 1. LINEAR REGRESSION
    # =========================
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # =========================
    # 2. DECISION TREE (TUNING)
    # =========================
    grid_dt = GridSearchCV(
        DecisionTreeRegressor(),
        {
            "max_depth":[3,5,10,None],
            "min_samples_split":[2,5,10]
        },
        cv=5
    )
    grid_dt.fit(X_train, y_train)
    pred_dt = grid_dt.predict(X_test)

    # =========================
    # 3. RANDOM FOREST (TUNING)
    # =========================
    grid_rf = GridSearchCV(
        RandomForestRegressor(),
        {
            "n_estimators":[100,200],
            "max_depth":[None,10],
            "min_samples_split":[2,5]
        },
        cv=5
    )
    grid_rf.fit(X_train, y_train)
    pred_rf = grid_rf.predict(X_test)

    # ======================================
    # EVALUASI
    # ======================================
    def evaluate(y_test, pred):
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        return mae, mse, rmse, r2

    print("Linear Regression:", evaluate(y_test, pred_lr))
    print("Decision Tree:", evaluate(y_test, pred_dt))
    print("Random Forest:", evaluate(y_test, pred_rf))

    results.append({
        "Split": name,
        "LR_R2": r2_score(y_test, pred_lr),
        "DT_R2": r2_score(y_test, pred_dt),
        "RF_R2": r2_score(y_test, pred_rf)
    })

# ==============================================
# MODEL COMPARISON
# ==============================================
df = pd.DataFrame(results)
print("\n=== PERBANDINGAN MODEL ===")
print(df)

df.set_index("Split").plot(kind="bar")
plt.title("Perbandingan R2 Score")
plt.ylabel("R2 Score")
plt.show()