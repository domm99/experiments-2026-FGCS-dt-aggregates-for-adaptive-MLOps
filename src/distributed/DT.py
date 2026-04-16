import pandas as pd
from torch import nn
from datetime import datetime
from src.distributed.DTAggregate import DTAggregate

class DT:

    def __init__(self, mid: str, data_path: str):
        self.mid = mid
        self._model = None
        self._dt_aggregate = None
        self.data = pd.read_csv(f'{data_path}/{mid}.csv')
        self.activation_date, self.stopping_date = self.__get_activation_interval()

    def is_active(self, time: pd.Timestamp) -> bool:
        return self.activation_date <= time <= self.stopping_date

    def __get_activation_interval(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        self.data['timestamp'] = pd.to_datetime(
            self.data['Measurement_date'] + ' ' + self.data['Measurement_time']
        )
        ts = self.data['timestamp']
        return ts.min(), ts.max()

    @model.setter
    def model(self, model: nn.Module):
        self._model = model

    @dt_aggregate.setter
    def dt_aggregate(self, dt: DTAggregate):
        self._dt_aggregate = dt

    ## TODO - implement this shit

    def inference(self):
        pass

    def test_loader_from_data(self):
        pass

    def export_test_metrics(self):
        pass