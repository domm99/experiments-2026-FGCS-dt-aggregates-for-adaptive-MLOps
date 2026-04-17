import glob
import pandas as pd
from src.distributed.Simulator import Simulator, Event
from src.distributed.utils import seed_everything
from src.distributed.LearningConfig import LearningConfig

def load_patients(data_folder: str) -> tuple[list[dict], pd.Timestamp, pd.Timestamp]:
    patients = []
    global_min, global_max = None, None

    for patient in glob.glob(f'{data_folder}/*.csv'):
        df = pd.read_csv(patient)
        df['timestamp'] = pd.to_datetime(
            df['Measurement_date'] + ' ' + df['Measurement_time']
        )
        ts = df['timestamp']
        min_time, max_time = ts.min(), ts.max()
        patients.append({'data': df, 'min_time': min_time, 'max_time': max_time})
        if min_time is None and max_time is None:
            global_min = min_time
            global_max = max_time
        else:
            if min_time < global_min:
                global_min = min_time
            if max_time > global_max:
                global_max = max_time
    return patients, global_min, global_max

if __name__ == "__main__":

    config = LearningConfig()
    data_folder = 'T1DiabetesGranada/split'
    seeds = [0]

    for seed in seeds:
        seed_everything(seed)

        all_patients, min_time, max_time = load_patients(data_folder)
        simulator = Simulator(data_folder, min_time, config, seed)

        for patient in all_patients:
            event = Event(
                time = patient['min_time'],
                priority = 0,
                event_type = 'PATIENT_BECOME_ACTIVE',
                payload = patient,
            )
            simulator.schedule_event(event)