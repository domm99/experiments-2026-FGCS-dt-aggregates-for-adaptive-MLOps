import glob
import heapq
import pandas as pd
from typing import Callable
from src.distributed.DT import DT
from dataclasses import dataclass, field
from src.distributed.DTAggregate import DTAggregate
from src.distributed.LearningConfig import LearningConfig

@dataclass(order=True)
class Event:
    time: pd.Timestamp
    priority: int
    event_type: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)

class EventQueue:

    def __init__(self):
        self._heap = []

    def push(self, event: Event):
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def empty(self) -> bool:
        return len(self._heap) == 0

class SimulationState:
    def __init__(self):
        self.active_patients = set()
        self.local_dts = {}
        self.patient_configs = {}

class Simulator:

    def __init__(self, data_folder: str, starting_time: pd.Timestamp, ending_time: pd.Timestamp, config: LearningConfig, seed: int):
        self._queue = EventQueue()
        self.data_folder = data_folder
        self.seed = seed
        self._config = config
        self.time = starting_time
        self._ending_time = ending_time
        self._state = SimulationState()
        self._handlers = {
            'PATIENT_BECOME_ACTIVE': self.__handle_patient_becomes_active,
        }
        self._dt_aggregate = DTAggregate(config, seed)

    def init_dts(self):
        files = glob.glob(f'{self.data_folder}/*.csv')
        dts = []
        for file in files:
            dt_id = file.split('/')[-1].split('.')[0]
            new_dt = DT(dt_id, self.data_folder, self._config)
            dts.append(new_dt)
        return dts

    def schedule_event(self, event: Event) -> None:
        self._queue.push(event)

    def __dispatch(self, event: Event):
        self._handlers[event.event_type](event)

    def start(self):
        while not self._queue.empty():
            event = self._queue.pop()
            self.__dispatch(event)

    def __handle_patient_becomes_active(self, event: Event):
        patient_id = event.payload['patient_id']
        current_time = event.time

        if patient_id in self._state.active_patients:
            return

        if patient_id not in self._state.local_dts:
            self._state.local_dts[patient_id] = DT(patient_id, self.data_folder, self._config)

        local_dt = self._state.local_dts[patient_id]
        local_dt.activate(current_time)
        self._state.active_patients.add(patient_id)
        self._dt_aggregate.register_active_dt(local_dt, current_time) ## TODO Implement this
        local_dt.model = self._dt_aggregate.model

        first_forecast_time = self.forecast_policy.first_forecast_time(
            patient_id=patient_id,
            activation_time=current_time,
            local_dt=local_dt,
        ) ## TODO fix this porca madonna

        if first_forecast_time is not None and first_forecast_time <= self._ending_time:
            self.schedule_event(Event(
                time=first_forecast_time,
                priority=10,
                event_type="LOCAL_FORECAST",
                payload={"patient_id": patient_id}
            ))