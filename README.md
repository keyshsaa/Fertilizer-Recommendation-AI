# Fertilizer Recommendation Model

A machine learning project that recommends a suitable fertilizer from soil and environmental conditions, using a tuned XGBoost classifier.

## Overview
**Input features:** soil type, crop type, temperature, moisture, rainfall, pH, nitrogen, phosphorous, potassium, carbon.
**Output:** the recommended fertilizer plus a short explanation of when it is used.

The experiments were done in Python (Google Colab) and compare Decision Tree, Random Forest, and XGBoost before tuning the final model.

## Project Structure
```
├── Ensemble_TrialnError.ipynb   # Full experiment: EDA, baselines, tuning, evaluation, SHAP
├── src/
│   ├── train.py                 # Train, evaluate, and save the model
│   └── predict.py               # Get a recommendation from the saved model
├── models/                      # Saved model is generated here (not committed)
├── requirements.txt
└── README.md
```

## What's Included
- Data cleaning (drops rows with impossible negative values) and exploratory analysis
- Separate label encoders for each categorical column
- Baseline model comparison (bias/variance check)
- Hyperparameter tuning with GridSearchCV
- Stratified 5-fold cross validation
- Accuracy, classification report, and confusion matrix
- SHAP feature importance
- Recommendation function with explanations

## Results
| Evaluation | Accuracy |
|---|---|
| Validation set | 0.9893 |
| Stratified 5-fold CV (train) | 0.9933 ± 0.0047 |
| Test set | 0.9929 |

Best parameters: `learning_rate=0.2`, `max_depth=3`, `n_estimators=100`, `subsample=1.0`.
Split: 80% train / 10% validation / 10% test (stratified, `random_state=42`).

## Setup
```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
**Notebook:** open `Ensemble_TrialnError.ipynb` in Jupyter or Colab and run all cells.

**Scripts:**
```bash
python src/train.py                              # downloads data via kagglehub, trains, saves the model
python src/train.py --csv path/to/dataset.csv    # or use a local CSV

python src/predict.py --Temperature 25 --Moisture 0.7 --Rainfall 200 --PH 6.5 \
  --Nitrogen 70 --Phosphorous 60 --Potassium 80 --Carbon 1.2 \
  --Soil "Loamy Soil" --Crop rice
```
Downloading from Kaggle may require credentials (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`).

## Dataset
[Fertilizer Recommendation (Kaggle)](https://www.kaggle.com/datasets/nishchalchandel/fertilizer-recommendation): 3,100 rows, 2,798 after cleaning.
