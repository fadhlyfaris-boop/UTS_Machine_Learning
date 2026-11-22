# ============================================================
# UTS MACHINE LEARNING
# DATA PREPROCESSING
# WATER POTABILITY CLASSIFICATION
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# 1. KONFIGURASI
# ------------------------------------------------------------

DATASET_PATH = "dataset/water_potability.csv"

print("=" * 65)
print("             DATA PREPROCESSING")
print("             WATER POTABILITY DATASET")
print("=" * 65)

# ------------------------------------------------------------
# 2. MEMUAT DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("\n[1] INFORMASI DATASET")
print("-" * 65)

print(f"Jumlah Data Awal : {df.shape[0]}")
print(f"Jumlah Kolom     : {df.shape[1]}")

# ------------------------------------------------------------
# 3. MEMERIKSA DATA DUPLIKAT
# ------------------------------------------------------------

print("\n[2] PEMERIKSAAN DUPLIKAT")
print("-" * 65)

print(f"Data Duplikat: {df.duplicated().sum()}")

# ------------------------------------------------------------
# 4. MEMISAHKAN FITUR DAN TARGET
# ------------------------------------------------------------

X = df.drop(columns=["Potability"])
y = df["Potability"]

print("\n[3] PEMISAHAN FITUR DAN TARGET")
print("-" * 65)

print(f"Jumlah Fitur : {X.shape[1]}")
print(f"Jumlah Target: {y.shape[0]}")

# ------------------------------------------------------------
# 5. MEMBAGI DATA TRAINING DAN TESTING
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n[4] PEMBAGIAN DATASET")
print("-" * 65)

print(f"Data Training : {X_train.shape[0]}")
print(f"Data Testing  : {X_test.shape[0]}")

# ------------------------------------------------------------
# 6. MEMBUAT PIPELINE IMPUTASI
# ------------------------------------------------------------

imputer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Fit hanya pada data training
X_train_processed = imputer.fit_transform(X_train)

# Transform data testing menggunakan median training
X_test_processed = imputer.transform(X_test)

# Mengembalikan hasil ke DataFrame
X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=X.columns,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=X.columns,
    index=X_test.index
)

print("\n[5] PENANGANAN MISSING VALUES")
print("-" * 65)

print("Metode: Median Imputation")

print("\nMissing Values Training:")
print(X_train_processed.isnull().sum())

print("\nMissing Values Testing:")
print(X_test_processed.isnull().sum())

# ------------------------------------------------------------
# 7. MEMERIKSA DISTRIBUSI TARGET
# ------------------------------------------------------------

print("\n[6] DISTRIBUSI TARGET")
print("-" * 65)

print("Training:")
print(y_train.value_counts())

print("\nTesting:")
print(y_test.value_counts())

# ------------------------------------------------------------
# 8. MENYIMPAN HASIL PREPROCESSING
# ------------------------------------------------------------

X_train_processed.to_csv(
    "dataset/X_train.csv",
    index=False
)

X_test_processed.to_csv(
    "dataset/X_test.csv",
    index=False
)

y_train.to_csv(
    "dataset/y_train.csv",
    index=False
)

y_test.to_csv(
    "dataset/y_test.csv",
    index=False
)

print("\n[7] PENYIMPANAN DATA")
print("-" * 65)

print("File berhasil disimpan:")
print("- dataset/X_train.csv")
print("- dataset/X_test.csv")
print("- dataset/y_train.csv")
print("- dataset/y_test.csv")

print("\n" + "=" * 65)
print("       DATA PREPROCESSING BERHASIL")
print("=" * 65)

# Menyimpan objek imputer untuk deployment
joblib.dump(imputer, "model/imputer.pkl")

print("Imputer berhasil disimpan: model/imputer.pkl")