import os
import sys
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI

# Paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_weights", "lstm_best.pt")
CSV_PATH   = os.path.join(BASE_DIR, "data", "LSTM-Multivariate_pollution.csv")

sys.path.append(os.path.join(BASE_DIR, "src", "models"))
sys.path.append(os.path.join(BASE_DIR, "src"))

# Local imports
from schemas import PredictionRequest, PredictionResponse
from lstm_model import LSTMForecaster
from preprocessing import load_and_encode, split_data, fit_scaler

# App
app = FastAPI(title="PM2.5 Prediction API")

# Model
model = LSTMForecaster(input_size=11)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

# Scaler
df_full                      = load_and_encode(CSV_PATH)
train_df, val_df, test_df    = split_data(df_full)
scaler, _, _, _              = fit_scaler(train_df, val_df, test_df)

@app.get("/")
def root():
    return {"message": "Backend funcionando"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):

    wnd_dir_map = {"NE": [1,0,0,0], "NW": [0,1,0,0], "SE": [0,0,1,0], "cv": [0,0,0,1]}
    wnd_encoded = wnd_dir_map[data.wnd_dir]

    sequence = []
    for row in data.history:
        sequence.append(row + wnd_encoded) 

    # Normalizar
    sequence_scaled = scaler.transform(sequence) 
    x = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0) 
    
    with torch.no_grad():
        prediction = model(x)

    # Desnormalizar
    pred_scaled = prediction.squeeze().item()
    pred_value = scaler.inverse_transform(
        [[pred_scaled] + [0] * (scaler.n_features_in_ - 1)]
    )[0][0]

    return PredictionResponse(prediction=round(float(pred_value), 2))