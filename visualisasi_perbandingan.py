import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("=" * 65)
print("       VISUALISASI PERBANDINGAN RANDOM FOREST")
print("=" * 65)

# ==========================================================
# 1. MEMUAT DATA TESTING
# ==========================================================

X_test = pd.read_csv("dataset/X_test.csv")
y_test = pd.read_csv("dataset/y_test.csv").squeeze()

# ==========================================================
# 2. MEMUAT KETIGA MODEL
# ==========================================================

models = {
    "Baseline": joblib.load("model/random_forest_model.pkl"),
    "Optimized": joblib.load("model/random_forest_optimized.pkl"),
    "Safety Optimized": joblib.load("model/random_forest_safety.pkl")
}

# ==========================================================
# 3. EVALUASI KETIGA MODEL
# ==========================================================

results = []
error_results = []

print("\n[1] HASIL EVALUASI MODEL")
print("-" * 65)

for name, model in models.items():

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Recall Tidak Layak": recall_score(
            y_test, y_pred, pos_label=0
        ),
        "F1 Tidak Layak": f1_score(
            y_test, y_pred, pos_label=0
        )
    })

    error_results.append({
        "Model": name,
        "Air Tidak Layak Salah Diprediksi Layak": fp,
        "Air Layak Salah Diprediksi Tidak Layak": fn
    })

    print(f"\nModel: {name}")
    print(f"Accuracy              : {accuracy:.4f}")
    print(f"Precision Layak       : {precision:.4f}")
    print(f"Recall Layak          : {recall:.4f}")
    print(f"F1-Score Layak        : {f1:.4f}")
    print(f"Recall Tidak Layak    : {recall_score(y_test, y_pred, pos_label=0):.4f}")
    print(f"F1-Score Tidak Layak  : {f1_score(y_test, y_pred, pos_label=0):.4f}")
    print(f"False Safe            : {fp}")
    print(f"False Unsafe          : {fn}")

# ==========================================================
# 4. MEMBUAT DATAFRAME
# ==========================================================

df_results = pd.DataFrame(results)
df_errors = pd.DataFrame(error_results)

print("\n[2] TABEL PERBANDINGAN")
print("-" * 65)
print(df_results.round(4).to_string(index=False))

print("\n[3] PERBANDINGAN KESALAHAN")
print("-" * 65)
print(df_errors.to_string(index=False))

# ==========================================================
# 5. MENYIMPAN HASIL CSV
# ==========================================================

os.makedirs("images", exist_ok=True)

df_results.to_csv(
    "model/perbandingan_lengkap.csv",
    index=False
)

df_errors.to_csv(
    "model/perbandingan_kesalahan.csv",
    index=False
)

# ==========================================================
# 6. GRAFIK PERBANDINGAN METRIK
# ==========================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

plot_data = df_results.set_index("Model")[metrics]

ax = plot_data.plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1),
    width=0.8
)

plt.title("Perbandingan Performa Tiga Model Random Forest")
plt.xlabel("Model")
plt.ylabel("Nilai Evaluasi")
plt.xticks(rotation=0)
plt.legend(title="Metrik")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "images/perbandingan_performa.png",
    dpi=300
)

plt.close()

# ==========================================================
# 7. GRAFIK PERBANDINGAN KESALAHAN
# ==========================================================

error_plot = df_errors.set_index("Model")

ax = error_plot.plot(
    kind="bar",
    figsize=(11, 6),
    width=0.7
)

plt.title("Perbandingan Kesalahan Klasifikasi Tiga Model")
plt.xlabel("Model")
plt.ylabel("Jumlah Kesalahan")
plt.xticks(rotation=0)
plt.legend(title="Jenis Kesalahan")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "images/perbandingan_kesalahan.png",
    dpi=300
)

plt.close()

# ==========================================================
# 8. GRAFIK RECALL KELAS TIDAK LAYAK
# ==========================================================

plt.figure(figsize=(9, 5))

plt.bar(
    df_results["Model"],
    df_results["Recall Tidak Layak"]
)

plt.title("Perbandingan Recall Kelas Tidak Layak")
plt.xlabel("Model")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "images/perbandingan_recall_tidak_layak.png",
    dpi=300
)

plt.close()

print("\n[4] VISUALISASI BERHASIL DISIMPAN")
print("-" * 65)
print("images/perbandingan_performa.png")
print("images/perbandingan_kesalahan.png")
print("images/perbandingan_recall_tidak_layak.png")

print("\n[5] FILE CSV BERHASIL DISIMPAN")
print("-" * 65)
print("model/perbandingan_lengkap.csv")
print("model/perbandingan_kesalahan.csv")

print("\n" + "=" * 65)
print("       VISUALISASI PERBANDINGAN SELESAI")
print("=" * 65)