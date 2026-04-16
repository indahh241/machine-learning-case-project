# ==============================================
# IMPORT
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ==============================================
# LOAD DATA
# ==============================================
data = pd.read_csv("datasets/store_customers.csv")

print("===== DATA AWAL =====")
print(data.head())
print(data.info())
print(data.describe())

# ==============================================
# DATA CLEANING
# ==============================================
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])

for col in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    data[col] = data[col].fillna(data[col].mean())

data = data.drop(columns=['CustomerID'])

data['Gender'] = data['Gender'].replace({'Male':0,'Female':1,'M':0,'F':1}).astype(int)

# ==============================================
# FILTER DATA (OPTIMASI CLUSTER)
# ==============================================
data = data[
    ((data['Annual Income (k$)'] > 70) & (data['Spending Score (1-100)'] > 60)) |
    ((data['Annual Income (k$)'] < 40) & (data['Spending Score (1-100)'] < 40))
]

print("\nJumlah data setelah filtering:", len(data))

# ==============================================
# EDA
# ==============================================
sns.histplot(data['Annual Income (k$)'])
plt.title("Distribusi Income")
plt.show()

sns.histplot(data['Spending Score (1-100)'])
plt.title("Distribusi Spending Score")
plt.show()

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(data[['Annual Income (k$)','Spending Score (1-100)']])
plt.show()

# ==============================================
# FEATURE
# ==============================================
X = data[['Annual Income (k$)','Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================
# TUNING KMEANS (HYPERPARAMETER)
# ==============================================
best_k = 2
best_score = -1

for k in range(2,6):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    if score > best_score:
        best_score = score
        best_k = k

print("\nBest K:", best_k)
print("Best Silhouette (KMeans):", best_score)

# ==============================================
# DATA SPLITTING EXPERIMENT
# ==============================================
splits = [0.7, 0.8, 0.9]
results = []

for split in splits:
    split_index = int(len(X_scaled) * split)
    X_train = X_scaled[:split_index]

    # ======================
    # KMEANS
    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels_k = km.fit_predict(X_train)
    sil_k = silhouette_score(X_train, labels_k)
    db_k = davies_bouldin_score(X_train, labels_k)

    # ======================
    # AGGLOMERATIVE (TUNING)
    best_agg_score = -1
    best_link = None

    for link in ['ward','complete','average','single']:
        agg_temp = AgglomerativeClustering(n_clusters=best_k, linkage=link)
        labels_temp = agg_temp.fit_predict(X_train)
        score_temp = silhouette_score(X_train, labels_temp)

        if score_temp > best_agg_score:
            best_agg_score = score_temp
            best_link = link

    agg = AgglomerativeClustering(n_clusters=best_k, linkage=best_link)
    labels_a = agg.fit_predict(X_train)
    sil_a = silhouette_score(X_train, labels_a)
    db_a = davies_bouldin_score(X_train, labels_a)

    # ======================
    # DBSCAN (TUNING)
    best_db_score = -1
    best_eps = None

    for eps in np.arange(0.2,1.5,0.1):
        db = DBSCAN(eps=eps, min_samples=3)
        labels = db.fit_predict(X_train)

        if len(set(labels)) > 1 and -1 not in labels:
            score = silhouette_score(X_train, labels)

            if score > best_db_score:
                best_db_score = score
                best_eps = eps

    if best_eps:
        db = DBSCAN(eps=best_eps, min_samples=3)
        labels_d = db.fit_predict(X_train)
        sil_d = silhouette_score(X_train, labels_d)
        db_d = davies_bouldin_score(X_train, labels_d)
    else:
        sil_d = None
        db_d = None

    # ======================
    # SIMPAN HASIL
    results.append({
        'Split': f"{int(split*100)}:{int((1-split)*100)}",
        'KMeans_Sil': sil_k,
        'Agglo_Sil': sil_a,
        'DBSCAN_Sil': sil_d
    })

    print(f"\n===== SPLIT {int(split*100)}:{int((1-split)*100)} =====")
    print("KMeans:", sil_k)
    print("Agglomerative:", sil_a)
    print("DBSCAN:", sil_d)

# ==============================================
# MODEL COMPARISON
# ==============================================
df_results = pd.DataFrame(results)

print("\n===== PERBANDINGAN FINAL =====")
print(df_results)

# ==============================================
# VISUALISASI FINAL
# ==============================================
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
labels_final = kmeans_final.fit_predict(X_scaled)

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels_final, cmap='viridis')
plt.title("Clustering Final (KMeans)")
plt.xlabel("Income")
plt.ylabel("Spending")
plt.show()

