📊 Machine Learning Case Project – ILKOM 2025/2026
📌 Deskripsi Project

Project ini merupakan tugas kecil mata kuliah Machine Learning yang bertujuan untuk membangun dan membandingkan beberapa model Machine Learning menggunakan tiga pendekatan:

Klasifikasi
Regresi
Clustering

Setiap model dibangun melalui tahapan lengkap Machine Learning mulai dari data understanding hingga model evaluation.

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

1. Breast Cancer Dataset (Klasifikasi)
Jumlah data: 569
Target: diagnosis (0 = jinak, 1 = ganas)

Digunakan untuk memprediksi jenis kanker payudara
2. Energy Efficiency Dataset (Regresi)
Jumlah data: 768
Target: Heating_Load
Digunakan untuk memprediksi kebutuhan energi bangunan

3. Customer Dataset (Clustering)
Jumlah data: 1000
Fitur utama:
Age
Annual Income
Spending Score
Digunakan untuk segmentasi pelanggan

⚙️ Tahapan Machine Learning

Project ini mencakup tahapan lengkap:

Data Understanding
Data Preprocessing
Missing value handling
Encoding
Scaling
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
Model terbaik: KMeans
Silhouette Score: ~0.57
DBSCAN kurang optimal pada dataset ini

📊 Evaluasi Model

Klasifikasi:
Accuracy
Precision
Recall
F1-score
Confusion Matrix

Regresi:
MAE
MSE
RMSE
R² Score

Clustering:
Silhouette Score
Davies-Bouldin Index

🧠 Kesimpulan
Model Random Forest memberikan performa terbaik pada klasifikasi dan regresi
Model KMeans paling optimal untuk segmentasi pelanggan
DBSCAN kurang cocok untuk pola dataset yang digunakan

🚀 Cara Menjalankan Project

Install library:
pip install pandas numpy matplotlib seaborn scikit-learn

Jalankan masing-masing file:
python classification/breast_cancer_model.py
python regression/energy_efficiency_model.py
python clustering/store_customers_model.py


👩‍💻 Author

NAMA : Indah Haerunnisa
NIM  : F1G123045

📌 Catatan

Project ini dibuat untuk memenuhi tugas mata kuliah Machine Learning dan dapat dikembangkan lebih lanjut untuk implementasi berbasis web atau aplikasi.