from pydantic import BaseModel

class PredictionRequest(BaseModel):
    pm25: float
    dew: float
    temp: float
    press: float
    wnd_spd: float
    snow: float
    rain: float
    wnd_dir: str  

class PredictionResponse(BaseModel):
    prediction: float