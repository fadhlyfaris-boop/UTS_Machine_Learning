import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)

# =====================================================
# 1. MEMUAT MODEL DAN IMPUTER
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "random_forest_safety.pkl"
)

IMPUTER_PATH = os.path.join(
    BASE_DIR, "model", "imputer.pkl"
)

model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]


# =====================================================
# 2. VALIDASI INPUT
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    probability = None
    input_data = {}
    error = None

    if request.method == "POST":

        try:

            values = []

            for feature in FEATURES:

                raw_value = request.form.get(
                    feature, ""
                ).strip()

                # Menangani nilai kosong
                if raw_value == "":
                    values.append(np.nan)
                    continue

                value = float(raw_value)

                # Memastikan input berupa angka valid
                if not np.isfinite(value):
                    raise ValueError(
                        f"{feature} harus berupa angka yang valid."
                    )

                # Validasi pH
                if feature == "ph":

                    if value < 0 or value > 14:
                        raise ValueError(
                            "Nilai pH harus berada di antara 0 sampai 14."
                        )

                # Parameter lainnya tidak boleh negatif
                else:

                    if value < 0:
                        raise ValueError(
                            f"{feature} tidak boleh bernilai negatif."
                        )

                values.append(value)
                input_data[feature] = raw_value

            # Membuat DataFrame
            df_input = pd.DataFrame(
                [values],
                columns=FEATURES
            )

            # Imputasi nilai kosong
            df_processed = imputer.transform(df_input)

            # Prediksi model
            prediction = model.predict(df_processed)[0]

            probabilities = model.predict_proba(
                df_processed
            )[0]

            class_index = list(
                model.classes_
            ).index(prediction)

            probability = round(
                probabilities[class_index] * 100,
                2
            )

            # Hasil prediksi
            if prediction == 1:

                result = {
                    "status": "Layak",
                    "description": (
                        "Model memprediksi sampel air ini "
                        "termasuk kategori layak."
                    ),
                    "class": "safe"
                }

            else:

                result = {
                    "status": "Tidak Layak",
                    "description": (
                        "Model memprediksi sampel air ini "
                        "termasuk kategori tidak layak."
                    ),
                    "class": "unsafe"
                }

        except ValueError as e:

            error = str(e)

        except Exception as e:

            error = "Terjadi kesalahan saat melakukan prediksi."
            print("ERROR:", e)

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        input_data=input_data,
        error=error
    )


# =====================================================
# 3. MENJALANKAN APLIKASI
# =====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )