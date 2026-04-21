import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Clustering App", layout="wide")

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
}
.block-container {padding: 2rem 3rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #020617, #0f172a);}
h1 {
    text-align: center;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("""
<h1>✨ Customer Segmentation Dashboard ✨</h1>
<p style='text-align:center;'>Machine Learning Clustering Analysis</p>
""", unsafe_allow_html=True)

file = st.sidebar.file_uploader("📂 Upload Dataset CSV", type=["csv"])

# ===============================
# MAIN APP
# ===============================
if file:

    # LOAD DATA
    data = pd.read_csv(file)

    st.subheader("📋 Data Awal")
    st.dataframe(data.head())

    # DATASET OVERVIEW
    st.subheader("📌 Dataset Overview & Problem Definition")
    st.write(f"Jumlah Data: {data.shape[0]}")
    st.write(f"Jumlah Fitur: {data.shape[1]}")

    desc = pd.DataFrame({
        "Fitur": data.columns,
        "Tipe": data.dtypes.astype(str),
        "Missing": data.isnull().sum(),
        "Unique": data.nunique()
    })
    st.dataframe(desc)

    st.info("""
Dataset ini menggunakan metode **Unsupervised Learning (Clustering)**.

Tujuan:
- Segmentasi data
- Menemukan pola tersembunyi
- Mendukung pengambilan keputusan
""")

    # PREPROCESSING
    st.subheader("⚙️ Data Preprocessing")

    if data.isnull().sum().sum() > 0:
        data = data.dropna()
        st.warning("Missing value dihapus")

    for col in data.select_dtypes(include='object').columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if "id" not in col.lower()]

    if len(numeric_cols) < 2:
        st.error("Fitur numerik kurang dari 2")
        st.stop()

    st.success("Preprocessing selesai ✔️")

    # FEATURE SELECTION
    fitur = st.multiselect("🎯 Pilih fitur", numeric_cols, default=numeric_cols[:2])

    if len(fitur) < 2:
        st.warning("Pilih minimal 2 fitur")
        st.stop()

    X = data[fitur]

    # SCALING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.info("Data sudah di-scaling")

    # FILTER DATA
    st.subheader("🔍 Data Filtering (Optimasi Cluster)")

    if "Annual Income (k$)" in data.columns and "Spending Score (1-100)" in data.columns:
        filtered_data = data[
            ((data['Annual Income (k$)'] > 70) & (data['Spending Score (1-100)'] > 60)) |
            ((data['Annual Income (k$)'] < 40) & (data['Spending Score (1-100)'] < 40))
        ]
        st.write("Jumlah data setelah filtering:", len(filtered_data))
        st.dataframe(filtered_data.head())

    # EDA
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.plotly_chart(px.histogram(data, x=fitur[0]))
        st.caption("Distribusi fitur 1")

    with col_eda2:
        st.plotly_chart(px.histogram(data, x=fitur[1]))
        st.caption("Distribusi fitur 2")

    # CORRELATION
    st.subheader("📈 Correlation Matrix")
    corr = data[fitur].corr()
    st.plotly_chart(px.imshow(corr, text_auto=True))

    # HYPERPARAMETER
    st.subheader("⚙️ Hyperparameter Tuning (KMeans)")

    best_k = 2
    best_score = -1

    for k in range(2, 6):
        km = KMeans(n_clusters=k, n_init=20)
        labels = km.fit_predict(X_scaled)

        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k

    st.success(f"Best K: {best_k}")
    st.write(f"Best Silhouette Score: {round(best_score,4)}")

    # DATA SPLITTING
    st.subheader("📊 Data Splitting Experiment")

    splits = [0.7, 0.8, 0.9]
    results = []

    for split in splits:
        split_index = int(len(X_scaled) * split)
        X_train = X_scaled[:split_index]

        km = KMeans(n_clusters=best_k, n_init=20)
        labels = km.fit_predict(X_train)

        sil = silhouette_score(X_train, labels) if len(set(labels)) > 1 else 0

        results.append({
            "Split": f"{int(split*100)}:{int((1-split)*100)}",
            "Silhouette": sil
        })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results)
    st.plotly_chart(px.bar(df_results, x="Split", y="Silhouette"))

    # MODELING
    st.subheader("⚙️ Mode Pemodelan")
    mode = st.radio("Mode:", ["Auto (Rekomendasi)", "Manual"])

    model_results = {}

    for k in range(2, 6):
        model = KMeans(n_clusters=k, n_init=20)
        label = model.fit_predict(X_scaled)
        if len(set(label)) > 1:
            model_results[f"KMeans k={k}"] = (silhouette_score(X_scaled, label), label)

    for k in range(2, 6):
        model = AgglomerativeClustering(n_clusters=k)
        label = model.fit_predict(X_scaled)
        if len(set(label)) > 1:
            model_results[f"Agglomerative k={k}"] = (silhouette_score(X_scaled, label), label)

    model = DBSCAN(eps=0.5)
    label = model.fit_predict(X_scaled)
    if len(set(label)) > 1:
        model_results["DBSCAN"] = (silhouette_score(X_scaled, label), label)

    if len(model_results) == 0:
        st.error("Tidak ada model cocok")
        st.stop()

    # ELBOW
    st.subheader("📉 Elbow Method")

    inertia = []
    for k in range(1, 8):
        km = KMeans(n_clusters=k, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    st.plotly_chart(px.line(x=list(range(1,8)), y=inertia, markers=True))

    # BEST MODEL
    best_model = max(model_results, key=lambda x: model_results[x][0])
    best_score, best_labels = model_results[best_model]

    st.success(f"🏆 Model terbaik: {best_model}")

    # SELECT MODEL
    if mode == "Manual":
        chosen = st.selectbox("Pilih model:", list(model_results.keys()))
        score_used, labels = model_results[chosen]
        model_used = chosen
    else:
        score_used, labels = best_score, best_labels
        model_used = best_model

    st.info(f"Model aktif: {model_used}")

    # VISUAL
    st.subheader("📊 Visualisasi")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.scatter(x=X_scaled[:,0], y=X_scaled[:,1], color=labels.astype(str)))

    with col2:
        dist = pd.Series(labels).value_counts().reset_index()
        dist.columns = ["Cluster","Jumlah"]
        st.plotly_chart(px.bar(dist, x="Cluster", y="Jumlah"))

    # PCA
    st.subheader("🌌 PCA Visualization")

    pca = PCA(n_components=min(3, X_scaled.shape[1]))
    pca_result = pca.fit_transform(X_scaled)

    if pca_result.shape[1] >= 2:
        st.plotly_chart(px.scatter(x=pca_result[:,0], y=pca_result[:,1], color=labels.astype(str)))

    # INSIGHT
    st.subheader("🧠 Insight")

    if score_used < 0.5:
        st.error("Cluster buruk")
    elif score_used < 0.7:
        st.warning("Cluster cukup")
    else:
        st.success("Cluster sangat baik")

    # PROFILING
    data['Cluster'] = labels
    profile = data.groupby('Cluster')[fitur].mean()
    st.dataframe(profile)

    # DOWNLOAD
    st.download_button("📥 Download Hasil", data.to_csv(index=False), "hasil_cluster.csv")

else:
    st.info("📂 Upload dataset terlebih dahulu")