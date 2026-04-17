import pandas as pd
from torch import nn
from datetime import datetime
from src.distributed.DTAggregate import DTAggregate
from src.distributed.LearningConfig import LearningConfig
from src.distributed.utils import load_patient_series, PatientSeries

class DT:

    def __init__(self, mid: str, data_path: str, config: LearningConfig):
        self._mid = mid
        self._model = None
        self._config = config
        self._dt_aggregate = None
        self._data = pd.read_csv(f'{data_path}/{mid}.csv')
        self._activation_date, self._stopping_date = self.__get_activation_interval()

    def is_active(self, time: pd.Timestamp) -> bool:
        return self._activation_date <= time <= self._stopping_date

    def __get_activation_interval(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        self._data['timestamp'] = pd.to_datetime(
            self._data['Measurement_date'] + ' ' + self._data['Measurement_time']
        )
        ts = self._data['timestamp']
        return ts.min(), ts.max()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: nn.Module):
        self._model = model

    @property
    def dt_aggregate(self):
        return self._dt_aggregate

    @dt_aggregate.setter
    def dt_aggregate(self, dt: DTAggregate):
        self._dt_aggregate = dt

    def __get_patient_series(self, current_time: pd.Timestamp) -> PatientSeries:
        filtered_df = self._data[self._data['timestamp'] <= current_time]
        series = load_patient_series(
            patient_id=self._mid,
            patient_dataframe=filtered_df,
            sequence_length=self._config.sequence_length,
            prediction_horizon=self._config.prediction_horizon,
            train_ratio=self._config.train_ratio,
        )
        return series

    def notify_data_to_dt_aggregate(self, current_time: pd.Timestamp) -> None:
        my_series = self.__get_patient_series(current_time)
        self._dt_aggregate.notify_new_data(self._mid, my_series)

    ## TODO - implement this shit

    def inference(self):
        pass

    def test_loader_from_data(self):
        pass

    def export_test_metrics(self):
        pass