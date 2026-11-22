# KLASIFIKASI POTABILITAS AIR MENGGUNAKAN RANDOM FOREST

## Deskripsi Proyek

Proyek ini merupakan implementasi Machine Learning untuk mengklasifikasikan potabilitas air berdasarkan parameter fisikokimia menggunakan algoritma Random Forest Classifier.

Aplikasi dikembangkan menggunakan Python dan Flask sehingga pengguna dapat memasukkan parameter kualitas air dan memperoleh hasil prediksi apakah sampel air termasuk kategori layak atau tidak layak.

Proyek ini dibuat sebagai bagian dari Ujian Tengah Semester (UTS) mata kuliah Pembelajaran Mesin.

## Tujuan

1. Menganalisis karakteristik dataset potabilitas air.
2. Melakukan preprocessing dan penanganan missing values.
3. Membangun model klasifikasi menggunakan Random Forest.
4. Melakukan optimasi hyperparameter menggunakan GridSearchCV.
5. Membandingkan performa beberapa konfigurasi model.
6. Mengimplementasikan model ke dalam aplikasi web berbasis Flask.

## Dataset

Dataset yang digunakan adalah Water Potability dengan 3.276 baris data dan 10 kolom.

Dataset terdiri dari sembilan parameter kualitas air dan satu variabel target.

### Fitur

- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

### Target

- 0 = Tidak Layak
- 1 = Layak

## Metode Penelitian

Tahapan pengembangan meliputi:

1. Exploratory Data Analysis (EDA).
2. Data preprocessing.
3. Penanganan missing values menggunakan median imputation.
4. Pembagian dataset menggunakan rasio 80:20.
5. Pelatihan model Random Forest.
6. Optimasi hyperparameter menggunakan GridSearchCV.
7. Evaluasi menggunakan Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.
8. Implementasi model ke aplikasi web Flask.

## Hasil Evaluasi Model

Tiga konfigurasi model dibandingkan menggunakan data pengujian yang sama.

| Model | Accuracy | Recall Layak | Recall Tidak Layak |
|---|---:|---:|---:|
| Baseline | 66,77% | 32,42% | 88,75% |
| Optimized | 67,07% | 41,02% | 83,75% |
| Safety Optimized | 66,31% | 20,70% | 95,50% |

Model Safety Optimized dipilih sebagai model aplikasi dengan mempertimbangkan prioritas mendeteksi sampel air yang tidak layak.

Model tersebut memperoleh recall sebesar 95,50% untuk kelas Tidak Layak, dengan 18 sampel tidak layak yang salah diprediksi sebagai layak pada data pengujian.

Pemilihan model ini memiliki konsekuensi berupa penurunan kemampuan mendeteksi sampel air yang benar-benar layak.

## Teknologi yang Digunakan

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Git dan GitHub

## Struktur Proyek

```text
UTS_Machine_Learning/
│
├── dataset/
│   ├── water_potability.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── images/
│   ├── distribusi_potability.png
│   ├── distribusi_parameter.png
│   ├── missing_values.png
│   ├── matriks_korelasi.png
│   ├── feature_importance.png
│   └── perbandingan_performa.png
│
├── model/
│   ├── imputer.pkl
│   ├── random_forest_model.pkl
│   ├── random_forest_optimized.pkl
│   └── random_forest_safety.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── eda.py
├── preprocessing.py
├── train_model.py
├── optimasi_model.py
├── evaluasi_keamanan.py
├── visualisasi_perbandingan.py
├── requirements.txt
└── README.md
```

## Instalasi dan Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone https://github.com/fadhlyfaris-boop/UTS_Machine_Learning.git
```

### 2. Masuk ke Folder Proyek

```bash
cd UTS_Machine_Learning
```

### 3. Instalasi Library

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
python app.py
```

### 5. Buka Aplikasi

Akses melalui browser:

```text
http://127.0.0.1:5000
```

## Catatan Penting

Aplikasi ini merupakan sistem prediksi berbasis Machine Learning yang menggunakan dataset penelitian.

Hasil klasifikasi tidak dapat dijadikan pengganti pengujian laboratorium atau standar resmi kelayakan air minum. Keputusan mengenai keamanan air tetap memerlukan pemeriksaan kualitas air secara langsung.

## Pengembang

Muhammad Faris

Program Studi Teknik Informatika  
Universitas Bale Bandung

## Repository

GitHub: https://github.com/fadhlyfaris-boop/UTS_Machine_Learning
