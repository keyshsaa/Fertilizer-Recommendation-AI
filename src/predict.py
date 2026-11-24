"""Rekomendasi pupuk dari model yang sudah dilatih (jalankan src/train.py dulu).

Contoh:
    python src/predict.py --Temperature 25 --Moisture 0.7 --Rainfall 200 --PH 6.5 \
        --Nitrogen 70 --Phosphorous 60 --Potassium 80 --Carbon 1.2 \
        --Soil "Loamy Soil" --Crop rice
"""
import argparse

import pandas as pd
from joblib import load

NUMERIC = ["Temperature", "Moisture", "Rainfall", "PH", "Nitrogen", "Phosphorous", "Potassium", "Carbon"]


def rekomendasi_pupuk(data_input, model_path="models/fertilizer_model.joblib"):
    """Kembalikan (nama_pupuk, penjelasan) dari dict kondisi lahan."""
    art = load(model_path)
    df = pd.DataFrame([data_input])
    df["Soil"] = art["soil_encoder"].transform(df["Soil"])
    df["Crop"] = art["crop_encoder"].transform(df["Crop"])
    df = df[art["feature_order"]]
    pupuk = art["target_encoder"].inverse_transform(art["model"].predict(df))[0]
    return pupuk, art["mapping_remark"].get(pupuk, "Tidak ada keterangan.")


def main():
    ap = argparse.ArgumentParser()
    for col in NUMERIC:
        ap.add_argument(f"--{col}", type=float, required=True)
    ap.add_argument("--Soil", required=True)
    ap.add_argument("--Crop", required=True)
    ap.add_argument("--model", default="models/fertilizer_model.joblib")
    a = vars(ap.parse_args())
    model_path = a.pop("model")
    pupuk, remark = rekomendasi_pupuk(a, model_path)
    print(f"Pupuk      : {pupuk}\nPenjelasan : {remark}")


if __name__ == "__main__":
    main()
