
# ============================================================
# UTS MACHINE LEARNING
# KLASIFIKASI KUALITAS AIR MENGGUNAKAN RANDOM FOREST
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. KONFIGURASI
# ------------------------------------------------------------

DATASET_PATH = "dataset/water_potability.csv"
OUTPUT_DIR = "images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("       EXPLORATORY DATA ANALYSIS (EDA)")
print("       WATER POTABILITY DATASET")
print("=" * 65)

# ------------------------------------------------------------
# 2. MEMUAT DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("\n[1] DATASET BERHASIL DIMUAT")
print("-" * 65)

print(f"Jumlah Baris   : {df.shape[0]}")
print(f"Jumlah Kolom   : {df.shape[1]}")

# ------------------------------------------------------------
# 3. MENAMPILKAN LIMA DATA PERTAMA
# ------------------------------------------------------------

print("\n[2] LIMA DATA PERTAMA")
print("-" * 65)

print(df.head())

# ------------------------------------------------------------
# 4. INFORMASI DATASET
# ------------------------------------------------------------

print("\n[3] INFORMASI DATASET")
print("-" * 65)

df.info()

# ------------------------------------------------------------
# 5. STATISTIK DESKRIPTIF
# ------------------------------------------------------------

print("\n[4] STATISTIK DESKRIPTIF")
print("-" * 65)

print(df.describe().T)

# ------------------------------------------------------------
# 6. PEMERIKSAAN MISSING VALUES
# ------------------------------------------------------------

print("\n[5] PEMERIKSAAN MISSING VALUES")
print("-" * 65)

missing = df.isnull().sum()

missing_table = pd.DataFrame({
    "Jumlah Missing": missing,
    "Persentase (%)": (missing / len(df) * 100).round(2)
})

print(missing_table)

# ------------------------------------------------------------
# 7. PEMERIKSAAN DATA DUPLIKAT
# ------------------------------------------------------------

print("\n[6] PEMERIKSAAN DATA DUPLIKAT")
print("-" * 65)

print(f"Jumlah Data Duplikat: {df.duplicated().sum()}")

# ------------------------------------------------------------
# 8. DISTRIBUSI TARGET POTABILITY
# ------------------------------------------------------------

print("\n[7] DISTRIBUSI TARGET POTABILITY")
print("-" * 65)

print(df["Potability"].value_counts())

print("\nPersentase Kelas:")
print(df["Potability"].value_counts(normalize=True) * 100)

# ------------------------------------------------------------
# 9. VISUALISASI DISTRIBUSI TARGET
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Potability")

plt.title("Distribusi Klasifikasi Potabilitas Air")
plt.xlabel("Potability (0 = Tidak Layak, 1 = Layak)")
plt.ylabel("Jumlah Data")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distribusi_potability.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 10. VISUALISASI MISSING VALUES
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.heatmap(df.isnull(), cbar=False, yticklabels=False)

plt.title("Visualisasi Missing Values Dataset")
plt.xlabel("Parameter Kualitas Air")
plt.ylabel("Data")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/missing_values.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 11. DISTRIBUSI SETIAP PARAMETER
# ------------------------------------------------------------

df.hist(figsize=(15, 12), bins=30)

plt.suptitle("Distribusi Parameter Kualitas Air", fontsize=16)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distribusi_parameter.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 12. KORELASI ANTAR PARAMETER
# ------------------------------------------------------------

plt.figure(figsize=(12, 9))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Matriks Korelasi Parameter Kualitas Air")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/matriks_korelasi.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 13. MENYIMPAN RINGKASAN DATASET
# ------------------------------------------------------------

print("\n[8] EDA SELESAI")
print("-" * 65)

print("Seluruh visualisasi berhasil disimpan di folder images/")

print("\nDaftar Gambar:")
print("- distribusi_potability.png")
print("- missing_values.png")
print("- distribusi_parameter.png")
print("- matriks_korelasi.png")

print("\n" + "=" * 65)
print("       PROSES EDA BERHASIL DISELESAIKAN")
print("=" * 65)