"""Training model rekomendasi pupuk (XGBoost + GridSearchCV).

Mengikuti alur di Ensemble_TrialnError.ipynb:
cleaning -> encoding -> split 80/10/10 -> tuning -> evaluasi -> simpan model.

Pemakaian:
    python src/train.py                 # unduh dataset dari Kaggle via kagglehub
    python src/train.py --csv path.csv  # pakai CSV lokal
"""
import argparse
import os

import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DATASET = "nishchalchandel/fertilizer-recommendation"
CSV_NAME = "fertilizer_recommendation_dataset.csv"
NON_NEGATIVE_COLS = ["Rainfall", "Phosphorous", "Potassium", "Carbon", "Moisture", "Nitrogen"]
PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 150],
    "subsample": [0.8, 1.0],
}


def load_data(csv_path=None):
    if csv_path is None:
        import kagglehub  # diimpor di sini agar mode --csv tidak butuh kagglehub

        csv_path = os.path.join(kagglehub.dataset_download(DATASET), CSV_NAME)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path CSV lokal (opsional)")
    ap.add_argument("--out", default="models", help="Folder output model")
    args = ap.parse_args()

    df_raw = load_data(args.csv)
    print(f"Dataset: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")

    # Mapping Fertilizer -> Remark disimpan sebelum kolom Remark dibuang
    mapping_remark = (
        df_raw[["Fertilizer", "Remark"]].drop_duplicates().set_index("Fertilizer")["Remark"].to_dict()
    )

    # Cleaning: buang baris dengan nilai negatif (tidak valid secara domain)
    df = df_raw[df_raw[NON_NEGATIVE_COLS].ge(0).all(axis=1)].copy()
    print(f"Setelah cleaning: {len(df)} baris ({len(df_raw) - len(df)} dihapus)")
    df = df.drop(columns="Remark")

    # Encoding: satu encoder per kolom kategorik
    soil_enc, crop_enc, target_enc = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df["Soil"] = soil_enc.fit_transform(df["Soil"].astype(str))
    df["Crop"] = crop_enc.fit_transform(df["Crop"].astype(str))
    df["Fertilizer"] = target_enc.fit_transform(df["Fertilizer"].astype(str))

    X, y = df.drop(columns="Fertilizer"), df["Fertilizer"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="mlogloss"),
        PARAM_GRID, cv=3, scoring="accuracy", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print("Parameter terbaik:", grid.best_params_)
    print(f"Akurasi CV (grid): {grid.best_score_:.4f}")

    print(f"Akurasi validasi: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(model, X_train, y_train, cv=skf)
    print(f"Stratified 5-Fold: mean={cv.mean():.4f} std={cv.std():.4f}")

    y_pred = model.predict(X_test)
    print(f"Akurasi test: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, labels=range(len(target_enc.classes_)),
                                target_names=target_enc.classes_, zero_division=0))

    os.makedirs(args.out, exist_ok=True)
    dump(
        {
            "model": model,
            "soil_encoder": soil_enc,
            "crop_encoder": crop_enc,
            "target_encoder": target_enc,
            "mapping_remark": mapping_remark,
            "feature_order": list(X.columns),
        },
        os.path.join(args.out, "fertilizer_model.joblib"),
    )
    print(f"Model disimpan di {args.out}/fertilizer_model.joblib")


if __name__ == "__main__":
    main()
