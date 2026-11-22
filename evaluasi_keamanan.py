import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer
)

print("=" * 65)
print("       OPTIMASI RANDOM FOREST BERDASARKAN KEAMANAN AIR")
print("=" * 65)

# ==========================================================
# 1. MEMUAT DATA
# ==========================================================

X_train = pd.read_csv("dataset/X_train.csv")
X_test = pd.read_csv("dataset/X_test.csv")

y_train = pd.read_csv("dataset/y_train.csv").squeeze()
y_test = pd.read_csv("dataset/y_test.csv").squeeze()

print("\n[1] DATA BERHASIL DIMUAT")
print("-" * 65)
print("Data Training:", X_train.shape)
print("Data Testing :", X_test.shape)

# ==========================================================
# 2. GRID SEARCH
# ==========================================================

print("\n[2] MEMULAI OPTIMASI MODEL")
print("-" * 65)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": [None, "balanced"]
}

# Fokus mendeteksi kelas 0 (Tidak Layak)
scorer = make_scorer(
    recall_score,
    pos_label=0
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ==========================================================
# 3. PARAMETER TERBAIK
# ==========================================================

print("\n[3] PARAMETER TERBAIK")
print("-" * 65)

print(grid_search.best_params_)

print("\nRecall Cross Validation kelas Tidak Layak:")
print(f"{grid_search.best_score_:.4f}")

# ==========================================================
# 4. EVALUASI MODEL
# ==========================================================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision_unsafe = precision_score(
    y_test,
    y_pred,
    pos_label=0
)

recall_unsafe = recall_score(
    y_test,
    y_pred,
    pos_label=0
)

f1_unsafe = f1_score(
    y_test,
    y_pred,
    pos_label=0
)

print("\n[4] HASIL EVALUASI MODEL")
print("-" * 65)

print(f"Accuracy                 : {accuracy:.4f}")
print(f"Precision Tidak Layak    : {precision_unsafe:.4f}")
print(f"Recall Tidak Layak       : {recall_unsafe:.4f}")
print(f"F1-Score Tidak Layak     : {f1_unsafe:.4f}")

# ==========================================================
# 5. CLASSIFICATION REPORT
# ==========================================================

print("\n[5] CLASSIFICATION REPORT")
print("-" * 65)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Tidak Layak", "Layak"]
    )
)

# ==========================================================
# 6. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\n[6] CONFUSION MATRIX")
print("-" * 65)
print(cm)

# ==========================================================
# 7. MENYIMPAN MODEL
# ==========================================================

os.makedirs("model", exist_ok=True)
os.makedirs("images", exist_ok=True)

joblib.dump(
    best_model,
    "model/random_forest_safety.pkl"
)

print("\n[7] MODEL BERHASIL DISIMPAN")
print("-" * 65)
print("model/random_forest_safety.pkl")

# ==========================================================
# 8. VISUALISASI CONFUSION MATRIX
# ==========================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Tidak Layak", "Layak"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix - Safety Optimized Random Forest")
plt.tight_layout()

plt.savefig(
    "images/confusion_matrix_safety.png",
    dpi=300
)

plt.close()

print("\n[8] VISUALISASI BERHASIL DISIMPAN")
print("-" * 65)
print("images/confusion_matrix_safety.png")

print("\n" + "=" * 65)
print("       EVALUASI KEAMANAN AIR SELESAI")
print("=" * 65)