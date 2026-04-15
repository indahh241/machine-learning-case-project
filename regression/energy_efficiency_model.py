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

print(data.head())
print(data.info())
print(data.describe())

# ==============================================
# EDA
# ==============================================
sns.heatmap(data.corr(), cmap="coolwarm")
plt.show()

# ==============================================
# FEATURE
# ==============================================
X = data.drop("Heating_Load",axis=1)
y = data["Heating_Load"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==============================================
# SPLIT
# ==============================================
splits = {"70:30":0.3,"80:20":0.2,"90:10":0.1}
results = []

for name,test_size in splits.items():

    print(f"\n=== {name} ===")

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    pred_lr = lr.predict(X_test)

    # Decision Tree (Tuning)
    grid_dt = GridSearchCV(
        DecisionTreeRegressor(),
        {"max_depth":[3,5,10,None]},
        cv=5
    )
    grid_dt.fit(X_train,y_train)
    pred_dt = grid_dt.predict(X_test)

    # Random Forest (Tuning)
    grid_rf = GridSearchCV(
        RandomForestRegressor(),
        {"n_estimators":[100,200],"max_depth":[None,10]},
        cv=5
    )
    grid_rf.fit(X_train,y_train)
    pred_rf = grid_rf.predict(X_test)

    # METRIK
    def evaluate(y_test,pred):
        mae = mean_absolute_error(y_test,pred)
        mse = mean_squared_error(y_test,pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test,pred)
        return mae,mse,rmse,r2

    print("LR:",evaluate(y_test,pred_lr))
    print("DT:",evaluate(y_test,pred_dt))
    print("RF:",evaluate(y_test,pred_rf))

    results.append({
        "Split":name,
        "LR_R2":r2_score(y_test,pred_lr),
        "DT_R2":r2_score(y_test,pred_dt),
        "RF_R2":r2_score(y_test,pred_rf)
    })

df = pd.DataFrame(results)
print(df)

df.set_index("Split").plot(kind="bar")
plt.show()