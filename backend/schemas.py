from __future__ import annotations
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    history: list
    wnd_dir: str

class PredictionResponse(BaseModel):
    prediction: float