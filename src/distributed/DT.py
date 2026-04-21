import glob
import torch
import pandas as pd
from collections import OrderedDict
from src.distributed.LearningConfig import LearningConfig
from src.distributed.utils import load_patient_series, load_test_patient_series, PatientSeries, ForecastLSTM, evaluate, create_test_loaders, normalize_series

class DT:

    def __init__(self, mid: str, data_path: str, config: LearningConfig, seed: int):
        self._mid = mid
        self._time = None
        self._seed = seed
        self._model = None
        self._last_std = 0.0
        self._last_mean = 0.0
        self._config = config
        self._is_active = False
        self._dt_aggregate = None
        self._data = self.__upload_data(data_path, mid)

    def activate(self, current_time: pd.Timestamp):
        self._is_active = True
        self._time = current_time

    def deactivate(self):
        self._is_active = False

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, data: tuple[OrderedDict[str, torch.Tensor], float, float]):
        model, mean, std = data
        fresh_model = ForecastLSTM(
            hidden_size=self._config.hidden_size,
            num_layers=self._config.layers,
            dropout=self._config.dropout,
        ).to(self._config.device)
        fresh_model.load_state_dict(model)
        self._model = fresh_model
        self._last_mean = mean
        self._last_std = std

    @property
    def dt_aggregate(self):
        return self._dt_aggregate

    @dt_aggregate.setter
    def dt_aggregate(self, dt):
        self._dt_aggregate = dt

    def __filter_data_by_time(self, current_time: pd.Timestamp, last_train_time: pd.Timestamp = None) -> pd.DataFrame:
        if last_train_time is None:
            return self._data[self._data['timestamp'] <= current_time]
        return self._data[
            (self._data['timestamp'] >= last_train_time) &
            (self._data['timestamp'] <= current_time)
        ]

    def __get_patient_series(self, current_time: pd.Timestamp, last_train_time: pd.Timestamp = None) -> PatientSeries:
        filtered_df = self.__filter_data_by_time(current_time, last_train_time)
        series = load_patient_series(
            patient_id=self._mid,
            patient_dataframe=filtered_df,
            sequence_length=self._config.sequence_length,
            prediction_horizon=self._config.prediction_horizon,
            train_ratio=self._config.train_ratio,
        )
        return series

    def get_data(self, current_time: pd.Timestamp) -> PatientSeries:
        my_series = self.__get_patient_series(current_time)
        return my_series

    def inference(self, current_time: pd.Timestamp, last_training_time: pd.Timestamp) -> pd.DataFrame:
        loader, num_points = self.__test_loader_from_data(current_time, last_training_time)
        if loader is None:
            self.__export_test_metrics(
                {
                    'status': 'skipped_short_test_window',
                    'num_points': num_points,
                    'mse': float('nan'),
                    'mae': float('nan'),
                    'rmse': float('nan'),
                },
                current_time,
            )
            return
        metrics = evaluate(self._model, loader, self._config.device, self._last_mean, self._last_std)
        metrics['status'] = 'evaluated'
        metrics['num_points'] = num_points
        self.__export_test_metrics(metrics, current_time)

    def __upload_data(self, data_path: str, mid: str) -> pd.DataFrame:
        data = pd.read_csv(f'{data_path}/{mid}.csv')
        data['timestamp'] = pd.to_datetime(
            data['Measurement_date'] + ' ' + data['Measurement_time']
        )
        return data

    def __test_loader_from_data(self, current_time: pd.Timestamp, last_training_time: pd.Timestamp):
        filtered_df = self.__filter_data_by_time(current_time, last_training_time)
        series = load_test_patient_series(
            patient_id=self._mid,
            patient_dataframe=filtered_df,
            sequence_length=self._config.sequence_length,
            prediction_horizon=self._config.prediction_horizon,
        )
        if series is None:
            return None, len(filtered_df)
        normalized_series = normalize_series(series, self._last_mean, self._last_std)
        loader = create_test_loaders(
            normalized_series,
            self._config.sequence_length,
            self._config.prediction_horizon,
            self._config.stride,
            self._config.batch_size)
        return loader, len(series.values)

    def __export_test_metrics(self, metrics: dict, current_time: pd.Timestamp):
        files = glob.glob(f'{self._config.data_export_path}/*{current_time}*.csv')
        metrics['dt_id'] = self._mid
        if len(files) == 0:
            metrics_df = pd.DataFrame([metrics])
        else:
            metrics_df = pd.read_csv(files[0])
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
        metrics_df.to_csv(f'{self._config.data_export_path}/test_{current_time}-seed_{self._seed}.csv', index=False)
