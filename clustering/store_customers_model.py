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

print(data.head())
print(data.info())
print(data.describe())

# ==============================================
# CLEANING
# ==============================================
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])

for col in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    data[col] = data[col].fillna(data[col].mean())

data = data.drop(columns=['CustomerID'])

data['Gender'] = data['Gender'].map({'M':0,'F':1})

# ==============================================
# EDA
# ==============================================
sns.pairplot(data[['Age','Annual Income (k$)','Spending Score (1-100)']])
plt.show()

# ==============================================
# FEATURE
# ==============================================
X = data[['Age','Annual Income (k$)','Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================
# TUNING K
# ==============================================
best_k = 2
best_score = -1

for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    if score > best_score:
        best_score = score
        best_k = k

print("Best K:",best_k)

# ==============================================
# MODEL
# ==============================================
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels_k = kmeans.fit_predict(X_scaled)

# Agglomerative tuning
best_link = None
best_agg_score = -1

for link in ['ward','complete','average']:
    agg = AgglomerativeClustering(n_clusters=best_k, linkage=link)
    labels = agg.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    if score > best_agg_score:
        best_agg_score = score
        best_link = link

agg = AgglomerativeClustering(n_clusters=best_k, linkage=best_link)
labels_a = agg.fit_predict(X_scaled)

# DBSCAN tuning
best_db_score = -1
best_eps = None

for eps in [0.3,0.5,0.7,0.9,1.1]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_scaled)

    if len(set(labels)) > 1 and -1 not in labels:
        score = silhouette_score(X_scaled, labels)

        if score > best_db_score:
            best_db_score = score
            best_eps = eps

if best_eps:
    db = DBSCAN(eps=best_eps, min_samples=5)
    labels_d = db.fit_predict(X_scaled)
else:
    labels_d = None

# ==============================================
# EVALUASI
# ==============================================
print("\nKMeans:",silhouette_score(X_scaled,labels_k))
print("Agglomerative:",silhouette_score(X_scaled,labels_a))

if labels_d is not None:
    print("DBSCAN:",silhouette_score(X_scaled,labels_d))
else:
    print("DBSCAN tidak optimal")

# ==============================================
# VISUALISASI
# ==============================================
plt.scatter(X_scaled[:,1],X_scaled[:,2],c=labels_k)
plt.title("KMeans Clustering")
plt.show()