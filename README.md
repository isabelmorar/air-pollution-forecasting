# Air Pollution Forecasting — PM2.5 Prediction with LSTM and Transformer

Hourly PM2.5 air pollution forecasting for Beijing using two deep learning architectures: a stacked LSTM and a Transformer encoder. Both models are trained on a multivariate time-series dataset and evaluated on a held-out test set.

## Project Structure

```
├── data/                                   # Raw dataset (CSV)
├── model_weights/                          # Saved model checkpoints (.pt)
├── src/
│   ├── preprocessing.py                    # Data loading, encoding, scaling, sequence creation
│   ├── dataset.py                          # PyTorch Dataset wrapper
│   ├── trainer.py                          # Training loop, evaluation, metrics
│   └── models/
│       ├── lstm_model.py                   # Stacked LSTM architecture
│       └── transformer_model.py            # Transformer encoder architecture
├── pollution_forecasting_results.ipynb     # Main notebook with EDA, training, evaluation, and results
└── requirements.txt
```

## Dataset

Hourly measurements from the US Embassy in Beijing (2010–2014), including PM2.5 concentration and meteorological variables: dew point, temperature, pressure, wind direction, wind speed, snowfall, and rainfall. 43,800 records, no missing values.

Source: [LSTM Datasets — Multivariate/Univariate on Kaggle](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate?resource=download)

## Setup

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Notebook

Open [`pollution_forecasting_results.ipynb`](pollution_forecasting_results.ipynb) and run all cells from top to bottom. 
The notebook covers EDA, preprocessing, training, evaluation, and model comparison. Pretrained weights are provided in `model_weights/`. To reproduce the full results, run all cells including training. To skip retraining, run all cells up to and including the model definition cells, then jump directly to the evaluation cells — the weights will be loaded from disk.

## Results Overview

| Model       | MAE (μg/m³) | RMSE (μg/m³) | MAPE (%) | R²     |
|-------------|-------------|--------------|----------|--------|
| LSTM        | 11.12       | 21.26        | 22.43    | 0.9273 |
| Transformer | 10.91       | 21.39        | 20.85    | 0.9264 |

Both models use a 24-hour sliding window to predict the next hour's PM2.5 value and explain over 92% of the variance on the held-out test set. The LSTM achieves better RMSE and R², while the Transformer edges out on MAE and MAPE. 
Full analysis and visualizations are included in the [main project notebook](pollution_forecasting_results.ipynb).

## AI Usage Statement

Claude (Anthropic) was used as a coding assistant during the development of this project. Its use included help with code structure, docstrings, and notebook documentation. All model design decisions, architecture choices, training configuration, and analysis were made by the author.