import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .torch_models import (
    LSTMForecast,
    GRUForecast,
    Seq2SeqForecast,
    AttentionSeq2Seq,
)

from datetime import timedelta


class FofecastDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            seq_len: int,
            target: str,
            save_collumns: bool = False,
            forecast_horizon: int = 1
        ):
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.target = target

        df = df.copy()
        source_cols = df.columns
        df_cols = [target] + [col for col in df.columns if col != target and save_collumns]
        # Преобразование даты в datetime если еще не преобразована
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        self.dates = df.index.copy()
        self.min_date = df.index.min()
        self.freq = self._infer_frequency(df.index)

        df = self._add_time_features(df)

        self.df = df

        self.df_cols = df_cols
        self.feature_cols = [col for col in df.columns if col not in source_cols]
        self.values = df[df_cols + self.feature_cols].values


    def _infer_frequency(self, date_index):
        """Определяет частоту временного ряда"""

        return date_index[1] - date_index[0]

    def _add_time_features(self, df):
        """Добавляет временные признаки"""

        df['date'] = df.index

        df['dow_sin'] = np.sin(2 * np.pi * df['date'].dt.weekday / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['date'].dt.weekday / 7)
        df['month_sin'] = np.sin(2 * np.pi * (df['date'].dt.month-1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['date'].dt.month-1) / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.day / 31)
        df['is_holiday'] = (df['date'].dt.weekday > 4).astype(int)

        if not self.freq.days:
            df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)
       
        df = df.drop('date', axis=1)

        return df

    def get_future_time_features(self, start_date, horizon):
        """
        Генерирует временные признаки для будущих дат
        """
        future_dates = pd.date_range(start=start_date, periods=horizon+1, freq=self.freq)[1:]
        future_df = pd.DataFrame(index=future_dates)
        future_df['date'] = future_df.index


        future_df = self._add_time_features(future_df)

        time_features = future_df[self.feature_cols].values

        return torch.tensor(time_features, dtype=torch.float32)

    def set_forecast_horizon(
        self,
        horizon
    ):
        self.forecast_horizon = horizon

    def __len__(self):
        return len(self.df) - self.seq_len - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x_encoder = self.values[idx:idx + self.seq_len]

        y_true = self.values[idx + self.seq_len:idx + self.seq_len + self.forecast_horizon, 0]

        start_date = self.dates[idx + self.seq_len - 1]
        future_time_features = self.get_future_time_features(start_date, self.forecast_horizon)

        return {
            'encoder_input': torch.tensor(x_encoder, dtype=torch.float32),
            'decoder_targets': torch.tensor(y_true, dtype=torch.float32),
            'future_time_features': future_time_features
        }
    
class TorchPreprocessor:
    """Класс предобработки данных для PyTorch, Аналогичный как для Sklearn"""
    def __init__(self,
            target: str,
            freq,
            seq_len: int,
            save_collumns: bool = False,
        ):
        self.target = target
        self.save_collumns = save_collumns
        self.seq_len = seq_len
        self.freq = freq

    def fit(self, df: pd.DataFrame):
        """Обучает препроцессор на данных"""
        df = df.copy()
        source_cols = df.columns
        self.df_cols = [self.target] + [col for col in df.columns if col != self.target and self.save_collumns]

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        self.min_date = df.index.min()
        self.freq = self._infer_frequency(df.index)

        df = self._add_time_features(df)

        self.feature_cols = [col for col in df.columns if col not in source_cols]

    def transform(self, df: pd.DataFrame, horison):
        """Применяет предобработку к данным, для прогнозирования"""
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = self._add_time_features(df)
        x_encoder = df[self.df_cols + self.feature_cols].values[-self.seq_len:]

        start_date = df.index[-1]
        future_time_features = self.get_future_time_features(start_date, horison)
        

        return {
            'encoder_input': torch.tensor(x_encoder, dtype=torch.float32),
            'future_time_features': future_time_features
        }

    def fit_transform(self, df: pd.DataFrame, horizon):
        """Обучает и применяет предобработку"""
        self.fit(df)
        return self.transform(df, horizon)

    def _infer_frequency(self, date_index):
        """Определяет частоту временного ряда"""
        return date_index[1] - date_index[0]

    def _add_time_features(self, df):
        """Добавляет временные признаки"""
        df['date'] = df.index

        df['dow_sin'] = np.sin(2 * np.pi * df['date'].dt.weekday / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['date'].dt.weekday / 7)
        df['month_sin'] = np.sin(2 * np.pi * (df['date'].dt.month-1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['date'].dt.month-1) / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.day / 31)
        df['is_holiday'] = (df['date'].dt.weekday > 4).astype(int)

        if not self.freq.days:
            df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)

        df = df.drop('date', axis=1)

        return df

    def get_future_time_features(self, start_date, horizon):
        """
        Генерирует временные признаки для будущих дат
        """
        future_dates = pd.date_range(start=start_date, periods=horizon+1, freq=self.freq)[1:]
        future_df = pd.DataFrame(index=future_dates)
        future_df['date'] = future_df.index

        future_df = self._add_time_features(future_df)

        time_features = future_df[self.feature_cols].values

        return torch.tensor(time_features, dtype=torch.float32)
        
    def set_forecast_horizon(
        self,
        horizon
    ):
        self.forecast_horizon = horizon
        

class TorchModel:
    def __init__(
        self,
        model,
        target,
        preprocessor,
    ):
        self.model = model
        self.target = target
        self.preprocessor = preprocessor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() 

    def forecast(self, df: pd.DataFrame, horizon):
        """
        Делает прогноз на horizon шагов вперед.
        """
        # Применяем предобработку к данным
        processed_data = self.preprocessor.fit_transform(df, horizon)

        encoder_input = processed_data['encoder_input'].unsqueeze(0).to(self.device)
        future_time_features = processed_data['future_time_features'].unsqueeze(0).to(self.device) 
        predictions = self.model(encoder_input, future_time_features)
    
        predictions = predictions.squeeze(0).cpu().detach().numpy()

        return predictions.reshape(-1,)
    
model_day_params = {
    "GRU": {
        "input_size": 8,
        "hidden_size" :128,
        "num_layers" : 1,
    },
    "LSTM": {
        "input_size": 8,
        "hidden_size" :128,
        "num_layers" : 1,
    },
    "SEQ2SEQ": {
        "input_size": 10,
        "hidden_size" :128,
        "num_layers" : 1,
        "time_feature_size" : 7,
    },
    "SEQ2SEQ_full": {
        "input_size": 13,
        "hidden_size" :128,
        "num_layers" : 1,
        "time_feature_size" : 7,
    },
    "SEQ2SEQ_attention": {
        "input_size": 10,
        "hidden_size" :128,
        "num_layers" : 1,
        "time_feature_size" : 7,
    },
    "SEQ2SEQ_attention_full": {       
        "input_size": 13,
        "hidden_size" :128,
        "num_layers" : 1,
        "time_feature_size" : 7,
    },
}

model_hour_params = { 
    "GRU": {
        "input_size": 10,
        "hidden_size" :128,
        "num_layers" : 2,
    },
    "LSTM": {
        "input_size": 10,
        "hidden_size" :128,
        "num_layers" : 2,
    },
    "SEQ2SEQ": {
        "input_size": 12,
        "hidden_size" :128,
        "num_layers" : 2,
        "time_feature_size" : 9,
    },
    "SEQ2SEQ_full": {
        "input_size": 15,
        "hidden_size" :128,
        "num_layers" : 2,
        "time_feature_size" : 9,
    },
    "SEQ2SEQ_attention": {
        "input_size": 12,
        "hidden_size" :128,
        "num_layers" : 2,
        "time_feature_size" : 9,
    },
    "SEQ2SEQ_attention_full": {       
        "input_size": 15,
        "hidden_size" :128,
        "num_layers" : 2,
        "time_feature_size" : 9,
    },
}

models_dict = {
    "GRU": GRUForecast,
    "LSTM": LSTMForecast,
    "SEQ2SEQ": Seq2SeqForecast,
    "SEQ2SEQ_full": Seq2SeqForecast,
    "SEQ2SEQ_attention": AttentionSeq2Seq,
    "SEQ2SEQ_attention_full": AttentionSeq2Seq,
}

task_dict = {
    "incoming" : "incoming_volume",
    "outgoing" : "outgoing_volume"
}
def __load_torch_models(
    dir,
    target,
    time_delta,
    seq_len,
):
    models = []
    for model_name in models_dict.keys():
        model_path = f"{dir}/{model_name}.pth"
        if not os.path.exists(model_path):
            print(f"Модель {model_name} не найдена по пути {model_path}. Пропускаем.")
            continue

        # Определяем параметры модели в зависимости от частоты
        if time_delta.days: # Дневная частота
            model_params = model_day_params.get(model_name)
            
        else: # Часовая частота
            model_params = model_hour_params.get(model_name)

        save_collumns = "SEQ2SEQ" in model_name
        if not model_params:
            print(f"Параметры для модели {model_name} с частотой {time_delta} не определены. Пропускаем.")
            continue

        # Создаем экземпляр модели
        model_class = models_dict[model_name]
        model_instance = model_class(**model_params)

        # Загружаем веса
        model_instance.load_state_dict(torch.load(model_path))

        preprocessor = TorchPreprocessor(
            target=task_dict[target],
            freq=time_delta,
            save_collumns=save_collumns,
            seq_len=seq_len
        )

        torch_model = TorchModel(
            model=model_instance,
            target=target,
            preprocessor=preprocessor,
        )

        models.append({
            "name" : model_name,
            "model" : torch_model,
            "target" : target,
            "frequency" : "day" if time_delta.days else "hour",
            "model_type" : "torch",
        })

    return models

def load_torch_models(
    dir,
):
    params = [
        (f"{dir}/inc_day", "incoming", timedelta(days=1), 14),
        (f"{dir}/inc_hour", "incoming", timedelta(hours=1), 48),
        (f"{dir}/out_day", "outgoing", timedelta(days=1), 14),
        (f"{dir}/out_hour", "outgoing", timedelta(hours=1), 48),
    ]

    models = []
    for param in params:
        models.extend(__load_torch_models(*param))

    return models