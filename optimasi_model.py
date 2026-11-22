# ============================================================
# UTS MACHINE LEARNING
# OPTIMASI RANDOM FOREST
# WATER POTABILITY CLASSIFICATION
# ============================================================

import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
# 1. MEMUAT DATASET
# ------------------------------------------------------------

print("=" * 65)
print("             RANDOM FOREST OPTIMIZATION")
print("=" * 65)

X_train = pd.read_csv("dataset/X_train.csv")
X_test = pd.read_csv("dataset/X_test.csv")

y_train = pd.read_csv("dataset/y_train.csv").squeeze()
y_test = pd.read_csv("dataset/y_test.csv").squeeze()

# ------------------------------------------------------------
# 2. KONFIGURASI MODEL
# ------------------------------------------------------------

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# ------------------------------------------------------------
# 3. PARAMETER GRID
# ------------------------------------------------------------

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": [None, "balanced"]
}

# ------------------------------------------------------------
# 4. STRATIFIED CROSS VALIDATION
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# ------------------------------------------------------------
# 5. PROSES OPTIMASI
# ------------------------------------------------------------

print("\n[1] MEMULAI GRID SEARCH")
print("-" * 65)

grid_search.fit(X_train, y_train)

print("\n[2] PARAMETER TERBAIK")
print("-" * 65)

print(grid_search.best_params_)

print("\nF1-Score Cross Validation:")
print(round(grid_search.best_score_, 4))

# ------------------------------------------------------------
# 6. EVALUASI PADA DATA TESTING
# ------------------------------------------------------------

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n[3] HASIL EVALUASI MODEL OPTIMASI")
print("-" * 65)

print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score  : {f1:.4f} ({f1*100:.2f}%)")

# ------------------------------------------------------------
# 7. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n[4] CLASSIFICATION REPORT")
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

print("\n[5] CONFUSION MATRIX")
print("-" * 65)

print(cm)

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Tidak Layak", "Layak"],
    yticklabels=["Tidak Layak", "Layak"]
)

plt.title("Confusion Matrix - Optimized Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.savefig(
    "images/confusion_matrix_optimized.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 9. MENYIMPAN MODEL TERBAIK
# ------------------------------------------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(
    best_model,
    "model/random_forest_optimized.pkl"
)

print("\n[6] MODEL OPTIMASI BERHASIL DISIMPAN")
print("-" * 65)

print("model/random_forest_optimized.pkl")

# ------------------------------------------------------------
# 10. PERBANDINGAN BASELINE DAN OPTIMASI
# ------------------------------------------------------------

print("\n[7] PERBANDINGAN HASIL MODEL")
print("-" * 65)

baseline = {
    "Accuracy": 0.6677,
    "Precision": 0.6484,
    "Recall": 0.3242,
    "F1-Score": 0.4323
}

optimized = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}

comparison = pd.DataFrame(
    [baseline, optimized],
    index=["Baseline", "Optimized"]
)

print(comparison)

comparison.to_csv("model/perbandingan_model.csv")

print("\n" + "=" * 65)
print("             OPTIMASI SELESAI")
print("=" * 65)