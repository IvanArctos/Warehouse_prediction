from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import zscore
from pathlib import Path

from models import (
    load_arima_models,
    load_sklearn_models,
    load_torch_models,
    load_hybrid_models,
)


data_path = Path(__file__).resolve().parent.parent / "data"

def ts_from_csv(csv_path, resampling = "h"):
    df = pd.read_csv( data_path / csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df['date_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))

    df.set_index('date_time', inplace=True)
    df.drop(columns=['date', 'time'], inplace=True)
    if resampling == "12h":
        df = df.resample("12h", offset='9h').sum()
    else:
        df = df.resample(resampling).sum().fillna(0)
    for col in df.columns:
        df[col] = np.where(zscore(df[col]) > 3, df[col].mean(), df[col])

    return df

def load_data(
    csv_path,
    freq
):
    if isinstance(csv_path, str):
        df = ts_from_csv(csv_path, freq)
    else:
        dfs = [
            ts_from_csv(csv, freq) for csv in csv_path
        ]
        df = pd.concat(
            dfs,
            axis=1,
            join = 'inner'
        )
    max = df.max()
    for col in df.columns:
        df[col]/=df[col].max()

    return df, max

models_dir = Path(__file__).resolve().parent.parent / "models"

def lifespan(app: FastAPI):
    # Загрузка моделей
    app.state.models = (
        load_arima_models(models_dir) +
        load_sklearn_models(models_dir) +
        load_torch_models(models_dir) +
        load_hybrid_models(models_dir)
    )
    print("Models loaded successfully!")
    yield


app = FastAPI(lifespan=lifespan)

class ModelInfo(BaseModel):
    name: str
    target: str
    frequency: str
    model_type: str

class ForecastRequest(BaseModel):
    horizon: int
    target: Literal['incoming' , 'outgoing']
    frequency: Literal["day", "hour"] 
    model_type: Literal["arima", "sklearn", "torch", "hybrid"] 
    model_name: str 

class ForecastResponse(BaseModel):
    forecast: list[float]
    targets: list[float]
    dates: list[datetime]


    

@app.get("/models", response_model=list[ModelInfo])
async def get_models():
    """
    Endpoint для получения списка моделей.
    """
    return app.state.models



task_dict = {
    "outgoing" : "outgoing_volume",
    "incoming" : "incoming_volume",
}

data_dict = {
    "outgoing" : "outgoing.csv",
    "incoming" : "incoming.csv",
}

    
@app.post("/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    """
    Endpoint для получения прогноза.
    """
    model = next((
    model for model in app.state.models
    if (
        model["name"] == request.model_name and 
        model["target"]== request.target and 
        model["frequency"] == request.frequency and 
        model["model_type"] == request.model_type)
    ), None)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if "full" in model["name"]:
        data = data_dict.values()
    else:
        data = data_dict[request.target]

    idx = 400 if request.frequency=="day" else 400*24 
    prevous = 14 if request.frequency=="day" else 2*24 
    df, max = load_data(data, request.frequency[0])
    inital_data = df.iloc[:idx]
    target = task_dict[request.target]
    horizon = request.horizon
    preds = model["model"].forecast(inital_data, horizon) * max[target] / 10**6
    true = df[target].iloc[idx - prevous: idx + horizon]  * max[target] / 10**6
    # print(preds)
    
    return {
        "forecast" : preds.tolist(),
        "targets" : true.values.tolist(),
        "dates" : true.index
    }
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)