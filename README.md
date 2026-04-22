🚀 Machine Learning Case Project – ILKOM 2025/2026
<p align="center"> <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python"> <img src="https://img.shields.io/badge/Library-Scikit--Learn-orange?logo=scikitlearn"> <img src="https://img.shields.io/badge/Status-Completed-success"> <img src="https://img.shields.io/badge/Type-Machine%20Learning-purple"> </p>
📌 Deskripsi Project

Project ini merupakan tugas mata kuliah Machine Learning ILKOM 2025/2026 yang bertujuan untuk membangun dan membandingkan model Machine Learning menggunakan tiga pendekatan utama:

Klasifikasi
Regresi
Clustering

Selain itu, project ini juga dilengkapi dengan Web App berbasis Streamlit untuk melakukan analisis clustering secara interaktif.

🌐 Akses Web Aplikasi (Deployment)

Aplikasi dapat langsung diakses tanpa instalasi melalui link berikut:

👉 https://f1g123045c-clustering.streamlit.app/

⚙️ Cara Menggunakan Web
Upload dataset CSV
Pilih fitur yang ingin dianalisis
Pilih mode:
Auto (Rekomendasi Sistem) → sistem memilih model terbaik secara otomatis
Manual (Pilih Model Sendiri) → pengguna menentukan model yang digunakan
Lihat hasil analisis:
Model terbaik (berdasarkan Silhouette Score)
Hyperparameter tuning (penentuan K optimal)
Visualisasi cluster (scatter, distribusi, dll)
Elbow Method
PCA 2D & 3D visualization
Insight & rekomendasi otomatis
Cluster profiling
Download hasil clustering dalam bentuk CSV

📌 Catatan:
Web ini berjalan di cloud (Streamlit Cloud), sehingga:

Tidak perlu install Python
Bisa diakses dari perangkat apa saja
Cocok untuk demo/presentasi

📂 Struktur Project
machine-learning-case-project/
│
├── classification/
│   └── breast_cancer_model.py
│
├── regression/
│   └── energy_efficiency_model.py
│
├── clustering/
│   └── store_customers_model.py
│
├── datasets/
│   ├── breast_cancer.csv
│   ├── energy_efficiency.csv
│   └── store_customers.csv
│
├── app.py                  ← Web Streamlit
├── requirements.txt        ← Dependency
└── README.md

📊 Dataset yang Digunakan
🔬 Klasifikasi – Breast Cancer Dataset
Jumlah data: 569
Target: diagnosis (0 = jinak, 1 = ganas)
Tujuan: Prediksi kanker payudara

🏢 Regresi – Energy Efficiency Dataset
Jumlah data: 768
Target: Heating_Load
Tujuan: Prediksi kebutuhan energi

🛍️ Clustering – Customer Dataset
Data awal: 200
Setelah preprocessing: 59 data
Digunakan untuk segmentasi pelanggan berdasarkan:
Annual Income
Spending Score

⚙️ Tahapan Machine Learning
Project ini mengikuti pipeline standar Machine Learning:

Data Understanding
Data Cleaning (Missing Value Handling)
Encoding (LabelEncoder)
Feature Selection (menghapus ID yang tidak relevan)
Data Scaling (StandardScaler)
Exploratory Data Analysis (EDA)
Model Building
Model Evaluation
Model Comparison
Visualization & Insight

📌 Khusus Clustering:
Menggunakan Silhouette Score untuk evaluasi
Menggunakan Elbow Method untuk menentukan jumlah cluster optimal

🤖 Algoritma yang Digunakan
🔹 Klasifikasi
Logistic Regression
Support Vector Machine (SVM)
Random Forest

🔹 Regresi
Linear Regression
Decision Tree Regressor
Random Forest Regressor

🔹 Clustering
K-Means
Agglomerative Clustering
DBSCAN

📈 Hasil Model

✅ Klasifikasi
Akurasi tertinggi: 100% (SVM & Random Forest)
Semua model > 95%

✅ Regresi
R² Score tertinggi: ~0.99 (Random Forest)
Semua model > 0.96

✅ Clustering
Model terbaik: K-Means
Silhouette Score:
70:30 → 0.8024
80:20 → 0.7978
90:10 → 0.7874

📊 Evaluasi Model
1. Klasifikasi
Accuracy
Precision
Recall
F1-score
Confusion Matrix

2. Regresi
MAE
MSE
RMSE
R² Score

3. Clustering
Silhouette Score
Davies-Bouldin Index
🌌 Fitur Web Clustering

Web app memiliki fitur:

Upload dataset sendiri
Auto & Manual model selection
Data preprocessing otomatis (cleaning, encoding, scaling)
Feature selection interaktif
Data filtering (opsional untuk optimasi cluster)
Exploratory Data Analysis (EDA)
Correlation matrix visualization
Hyperparameter tuning otomatis (KMeans)
Data splitting experiment (evaluasi performa model)
Elbow Method visualization
PCA 2D & 3D visualization
Cluster profiling (rata-rata tiap cluster)
Insight & rekomendasi otomatis
Download hasil clustering

🧠 Kesimpulan
Random Forest terbaik untuk klasifikasi dan regresi
K-Means terbaik untuk clustering
Preprocessing dan scaling sangat berpengaruh
Web app membantu analisis data secara interaktif dan real-time

🚀 Cara Menjalankan (Local)

Jika ingin menjalankan secara lokal:

1. Install library
pip install -r requirements.txt
2. Jalankan Web App
streamlit run app.py


👩‍💻 Author

Nama: Indah Haerunnisa
NIM: F1G123045
Kelas: C 