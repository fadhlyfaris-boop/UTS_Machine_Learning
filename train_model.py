# ============================================================
# UTS MACHINE LEARNING
# RANDOM FOREST CLASSIFICATION
# WATER POTABILITY
# ============================================================

import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. KONFIGURASI
# ------------------------------------------------------------

os.makedirs("model", exist_ok=True)
os.makedirs("images", exist_ok=True)

print("=" * 65)
print("         RANDOM FOREST MODEL TRAINING")
print("         WATER POTABILITY CLASSIFICATION")
print("=" * 65)

# ------------------------------------------------------------
# 2. MEMUAT DATASET
# ------------------------------------------------------------

X_train = pd.read_csv("dataset/X_train.csv")
X_test = pd.read_csv("dataset/X_test.csv")

y_train = pd.read_csv("dataset/y_train.csv").squeeze()
y_test = pd.read_csv("dataset/y_test.csv").squeeze()

print("\n[1] DATASET BERHASIL DIMUAT")
print("-" * 65)

print(f"Data Training : {X_train.shape}")
print(f"Data Testing  : {X_test.shape}")

# ------------------------------------------------------------
# 3. MEMBANGUN MODEL RANDOM FOREST
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

print("\n[2] KONFIGURASI MODEL")
print("-" * 65)

print("Algoritma          : Random Forest")
print("Jumlah Pohon       : 200")
print("Max Depth          : None")
print("Min Samples Split  : 2")
print("Min Samples Leaf   : 1")
print("Random State       : 42")

# ------------------------------------------------------------
# 4. MELATIH MODEL
# ------------------------------------------------------------

print("\n[3] PROSES TRAINING")
print("-" * 65)

model.fit(X_train, y_train)

print("Model berhasil dilatih!")

# ------------------------------------------------------------
# 5. MELAKUKAN PREDIKSI
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("\n[4] PREDIKSI SELESAI")
print("-" * 65)

# ------------------------------------------------------------
# 6. EVALUASI MODEL
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n[5] HASIL EVALUASI MODEL")
print("-" * 65)

print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")

# ------------------------------------------------------------
# 7. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n[6] CLASSIFICATION REPORT")
print("-" * 65)

print(classification_report(
    y_test,
    y_pred,
    target_names=["Tidak Layak", "Layak"],
    zero_division=0
))

# ------------------------------------------------------------
# 8. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n[7] CONFUSION MATRIX")
print("-" * 65)

print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Tidak Layak", "Layak"],
    yticklabels=["Tidak Layak", "Layak"]
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 9. FEATURE IMPORTANCE
# ------------------------------------------------------------

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n[8] FEATURE IMPORTANCE")
print("-" * 65)

print(importance.to_string(index=False))

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Parameter")

plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 10. MENYIMPAN MODEL
# ------------------------------------------------------------

joblib.dump(model, "model/random_forest_model.pkl")

print("\n[9] MODEL BERHASIL DISIMPAN")
print("-" * 65)

print("Lokasi: model/random_forest_model.pkl")

# ------------------------------------------------------------
# 11. RINGKASAN
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("       RANDOM FOREST TRAINING SELESAI")
print("=" * 65)