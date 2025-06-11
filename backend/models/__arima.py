import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os
from datetime import datetime, timedelta


class ArimaModel:
    def __init__(
        self,
        saved_data,
        train_df,
    ):
        self.saved_data = saved_data
        self.train_df = train_df
        
    def get_model(self):
        model = SARIMAX(self.train_df, 
                            order=self.saved_data['order'],
                            seasonal_order=self.saved_data['seasonal_order'],
                            enforce_stationarity=False,
                            enforce_invertibility=False)
        model_fit = model.filter(self.saved_data['params'])

        return model_fit

    def forecast(self, df, horizon):
        start = len(df) + 1
        end = start + horizon
        
        model = self.get_model()
        print(start)
        predictions = model.get_prediction(start=start, end=end).predicted_mean
        return predictions.values
    

def load_arima_model(path, train_df):
    params = joblib.load(path)
    return ArimaModel(params, train_df)

arma_models = ["ARIMA", "SARIMA", "SARIMAX"]

task_dict = {
    "incoming" : "incoming_volume",
    "outgoing" : "outgoing_volume"
}

def __load_arima_models(
    dir,
    target,
    time_delta,
):
    models = []
    # train_df = pd.read_csv(os.path.join(dir, "train.csv"), parse_dates=True, index_col="date_time")
    train_df = pd.read_pickle(os.path.join(dir, "train.pkl"))

    for model_name in arma_models:
        path = os.path.join(dir, f"{model_name}.pkl")
        models.append({
            "model": load_arima_model(path, train_df,),
            "name": model_name,
            "target": target,
            "frequency": "day" if time_delta.days else "hour",
            "model_type": "arima"
        })
        
    return models
    

def load_arima_models(
    dir
):
    params = [
        (f"{dir}/inc_day", "incoming", timedelta(days=1)),
        (f"{dir}/inc_hour", "incoming", timedelta(hours=1)),
        (f"{dir}/out_day", "outgoing", timedelta(days=1)),
        (f"{dir}/out_hour", "outgoing", timedelta(hours=1)),
    ]

    models = []
    for param in params:
        models += __load_arima_models(*param)
    
    return models
