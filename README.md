# Car price prediction – and why no model beats the mean

Regression on a used-car dataset (`Price_USD`), with an honest negative result: **no model is reliably better than always predicting the average price**, and the notebook shows why.

**Data:** [Car Price Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/prince7489/car-price-prediction-dataset) – 1,000 rows, 12 columns.
Download `car_price_dataset_medium.csv` and put it next to the notebook.

## What is in the notebook
- Exploratory analysis: descriptive statistics, skewness, histograms
- Feature engineering: `Car_Age`, `Km_per_Year`, `Power_per_CC`
- Seven regression models in leak-free `Pipeline`s (scaling + one-hot encoding), compared with a mean baseline using MAE, RMSE, R² and 10-fold CV
- `GridSearchCV` with strongly regularised settings (Ridge, KNN, Random Forest)
- Diagnostics: correlation with the target, Kruskal–Wallis tests for categories, mutual information, and a **label-shuffling test**

## Results (test set, 330 cars)

| Model | RMSE (USD) | R² test | R² 10-fold CV |
|---|---|---|---|
| Ridge (alpha=10) | 33,357 | 0.011 | −0.044 |
| Linear Regression | 33,420 | 0.007 | −0.050 |
| **Baseline (mean)** | **33,542** | **−0.000** | **−0.016** |
| Random Forest | 34,231 | −0.042 | −0.087 |
| Gradient Boosting | 35,032 | −0.091 | −0.125 |
| Decision Tree (depth 3) | 35,180 | −0.100 | −0.071 |
| KNN (k=15) | 35,356 | −0.111 | −0.065 |
| Decision Tree (unrestricted) | 45,502 | −0.841 | −1.126 |

## Key findings
- All numeric features have |r| < 0.08 with the price; three of four categorical features show no price differences (`Owner_Type`, p = 0.013, does not survive a multiple-testing correction).
- Tuned and regularised models collapse towards the mean (CV R² ≈ −0.02); the tiny positive test R² of some models is within noise and not confirmed by cross-validation.
- A model trained on a **shuffled** price scores as badly as one trained on the real price.
- The dataset appears to be synthetic with a randomly generated price. The value of the project is the correct, leak-free methodology and the discipline of testing against a baseline and diagnosing a negative result.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook car_price_regression.ipynb
```
