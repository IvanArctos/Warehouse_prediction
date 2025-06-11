from datetime import timedelta
import joblib

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


class FeatureEngineer:

    def __init__(
        self,
        lag_config: dict[str,list[int]]=None,
        rolling_config: dict[str,list[int]]=None,
    ):
        self.lag_config = lag_config if lag_config else {}
        self.rolling_config = rolling_config if rolling_config else {}

        self.created_features = set()
        self.used_columns = set()
        self._fit = False

    def add_lags(self, df):
        """
        Добавляет лаги к выбранным колонкам.
        """
        df = df.copy()
        for col, lags in self.lag_config.items():
            if col not in df.columns:
                continue
            for lag in lags:
                new_col = f'{col}_lag{lag}'
                df[new_col] = df[col].shift(lag)
                self.created_features.add(new_col)
                self.used_columns.add(new_col)
        return df

    def add_rolling(self, df):
        """
        Добавляет скользящие средние к выбранным колонкам.
        """
        df = df.copy()
        for col, windows in self.rolling_config.items():
            if col not in df.columns:
                continue
            for window in windows:
                new_col = f'{col}_roll{window}'
                df[new_col] = df[col].shift().rolling(window=window, min_periods=1).mean()
                self.created_features.add(new_col)
                self.used_columns.add(new_col)
        return df

    def add_time_features(self, df):
        """
        Добавляет временные признаки к датафрейму на основе индекса.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Индекс датафрейма должен быть типа DatetimeIndex")

        # Создаем копию, чтобы не менять исходный датафрейм
        new_df = df.copy()


        # Базовые временные компоненты
        new_df[f'year'] = df.index.year
        new_df[f'month'] = df.index.month
        new_df[f'day'] = df.index.day
        new_df[f'dayofweek'] = df.index.dayofweek
        new_df[f'quarter'] = df.index.quarter
        # new_df[f'{pref}dayofyear'] = df.index.dayofyear

        # Добавляем номер недели в году
        # new_df[f'{pref}weekofyear'] = df.index.isocalendar().week

        # Временные компоненты (если в данных есть время)
        has_time = (df.index.hour > 0).any() or (df.index.minute > 0).any()
        if has_time:
            new_df[f'hour'] = df.index.hour

        # Флаги
        new_df[f'is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

        self.time_features = (set(new_df.columns) - set(df.columns))
        self.used_columns.update(self.time_features)
        return new_df


    def drop_unused_columns(self, df, keep_originals=True):
        """
        Удаляет колонки, которые не были задействованы в процессе создания признаков.
        """
        df = df.copy()
        originals = set(self.lag_config.keys()) | set(self.rolling_config.keys())
        if keep_originals:
            keep_cols = self.used_columns | originals
        else:
            keep_cols = self.used_columns
        drop_cols = set(df.columns) - keep_cols
        df = df.drop(columns=list(drop_cols))
        return df

    def fit_transform(self, df, keep_originals=True):
        """
        Выполняет полный цикл преобразований.
        """
        self.created_features = set()
        self.used_columns = set()
        df = self.add_lags(df)
        df = self.add_rolling(df)
        df = self.add_time_features(df)
        df = self.drop_unused_columns(df, keep_originals=keep_originals)
        self._fit = True
        return df

    def transform(self, df, keep_originals=True):
        """
        Применяет сохранённую конфигурацию к новым данным.
        """
        if not self._fit:
            raise RuntimeError("Сначала вызовите fit_transform на обучающей выборке!")
        return self.fit_transform(df, keep_originals=keep_originals)




class SklearnModel:
    def __init__(
        self,
        model,
        features,
        target,
        preprocessor,
        time_delta,

    ):
        self.model = model
        self.features = features
        self.target = target
        self.preprocessor = preprocessor
        self.model = model
        self.time_delta = time_delta


    def prepare_data(
        self,
        df : pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        df = df.copy()
        df = self.preprocessor.fit_transform(df)
        df = df.dropna()
        return df.drop(columns=self.target), df[self.target]

    def split_data(
        self,
        X,
        Y
    ):
        return train_test_split(X, Y, test_size=self.test_size, shuffle=False) 

    def forecast(
            self,
            initial_data,
            steps,
    ):
        current_data = initial_data.copy()
        forecasts = []

        for _ in range(steps):
            # Создаем копию данных для текущей итерации
            temp_df = current_data.copy()
            new_index = temp_df.index[-1] + self.time_delta
            new_row = {self.target: 0}
            temp_df.loc[new_index] = new_row

            # Создание фичей
            df, _ = self.prepare_data(temp_df)
            df = df[self.features]
            # Получаем последнюю строку для прогноза
            features = df.iloc[[-1]]

            # Прогнозируем
            pred = self.model.predict(features)[0]
            forecasts.append(pred )

            # Обновляем данные
            new_row = {self.target: pred}
            current_data.loc[new_index] = new_row

        return np.array(forecasts)
target_dict = {
    "incoming" : "incoming_volume",
    "outgoing" : "outgoing_volume",
}


def load_models(
    dir,
    target,
    time_delta,
):
    with open(os.path.join(dir, 'features.json'), 'r') as f:
        features = json.load(f)
    if time_delta.days or "12h" in target:
        preprocessor = FeatureEngineer(
            lag_config={
                target_dict[target] : range(1,15)
            },
            rolling_config={
                target_dict[target] : [2,7,14,30]
            }
        )
    else:
        preprocessor = FeatureEngineer(
            lag_config={
                target_dict[target] : range(1,24)
            },
            rolling_config={
                target_dict[target] : [12,24,48,24*7]
            }
        )
        
    models = []
    for model_name, features in features.items():
        model_path = os.path.join(dir, f"{model_name}.pkl")
        model_ = joblib.load(model_path)
        model = SklearnModel(
            model=model_,
            features=features,
            target=target_dict[target],
            preprocessor=preprocessor,
            time_delta=time_delta,
        )
        models.append({
            "name" : model_name,
            "model" : model,
            "target" : target,
            "frequency" : "day" if time_delta.days else "hour",
            "model_type" : "sklearn",
        })

    return models

def load_sklearn_models(
    dir,
):
    params = [
        (f"{dir}/inc_day", "incoming", timedelta(days=1)),
        (f"{dir}/inc_hour", "incoming", timedelta(hours=1)),
        (f"{dir}/out_day", "outgoing", timedelta(days=1)),
        (f"{dir}/out_hour", "outgoing", timedelta(hours=1)),
    ]

    models = []
    for param in params:
        models.extend(load_models(*param))

    return models