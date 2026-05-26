import os
import sys
import torch
import numpy as np
from fastapi import FastAPI
from schemas import PredictionRequest, PredictionResponse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "models"))

from lstm_model import LSTMForecaster

app = FastAPI(title="PM2.5 Prediction API")

MODEL_PATH = os.path.join(BASE_DIR, "model_weights", "lstm_best.pt")
model = LSTMForecaster(input_size=11)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

# Endpoint health
@app.get("/")
def root():
    return {"message": "Backend funcionando"}

# Endpoint predicción
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):

    features = np.array([
        data.pm25,
        data.dew,
        data.temp,
        data.press,
        data.wnd_spd,
        data.snow,
        data.rain
    ], dtype=np.float32)

    wnd_dir_map = {"NE": [1,0,0,0], "NW": [0,1,0,0], "SE": [0,0,1,0], "cv": [0,0,0,1]}
    features = np.array([
        data.pm25, data.dew, data.temp, data.press,
        data.wnd_spd, data.snow, data.rain,
        *wnd_dir_map[data.wnd_dir]
    ], dtype=np.float32)

    x = torch.tensor(features).unsqueeze(0).unsqueeze(0) 

    with torch.no_grad():
        prediction = model(x)

    pred_value = float(prediction.squeeze().item())

    return PredictionResponse(
        prediction=round(pred_value, 2)
    )