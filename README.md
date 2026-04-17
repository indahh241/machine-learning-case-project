🚀 Machine Learning Case Project – ILKOM 2025/2026
<p align="center"> <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python"> <img src="https://img.shields.io/badge/Library-Scikit--Learn-orange?logo=scikitlearn"> <img src="https://img.shields.io/badge/Status-Completed-success"> <img src="https://img.shields.io/badge/Type-Machine%20Learning-purple"> </p>

📌 Deskripsi Project

Project ini merupakan tugas mata kuliah Machine Learning ILKOM 2025/2026 yang bertujuan untuk membangun dan membandingkan model Machine Learning menggunakan tiga pendekatan:

Klasifikasi
Regresi
Clustering

Setiap model dibangun melalui tahapan lengkap Machine Learning.

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
Data Understanding
Data Preprocessing
Exploratory Data Analysis (EDA)
Data Splitting (70:30, 80:20, 90:10)
Model Building
Hyperparameter Tuning
Model Evaluation
Model Comparison

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

Agglomerative dan DBSCAN menghasilkan performa yang setara.

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

🧠 Kesimpulan
Random Forest terbaik untuk klasifikasi dan regresi
K-Means terbaik untuk clustering
Preprocessing sangat berpengaruh terhadap hasil

🚀 Cara Menjalankan
Install library
pip install pandas numpy matplotlib seaborn scikit-learn

Jalankan program
python classification/breast_cancer_model.py
python regression/energy_efficiency_model.py
python clustering/store_customers_model.py

👩‍💻 Author
Indah Haerunnisa
NIM: F1G123045