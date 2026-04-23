import glob
import pandas as pd
from pathlib import Path
from codecarbon import track_emissions
from src.distributed.utils import seed_everything
from src.distributed.Simulator import Simulator, Event
from src.distributed.LearningConfig import LearningConfig
from src.distributed.Monitors import ActivationPatientsMonitor, PerformanceDriftMonitor


def load_patients(data_folder: str) -> tuple[list[dict], pd.Timestamp, pd.Timestamp]:
    patients = []
    global_min, global_max = None, None

    for patient in glob.glob(f'{data_folder}/*.csv'):
        df = pd.read_csv(patient)
        df['timestamp'] = pd.to_datetime(
            df['Measurement_date'] + ' ' + df['Measurement_time']
        )
        patient_id = df['Patient_ID'].iloc[0]
        ts = df['timestamp']
        min_time, max_time = ts.min(), ts.max()
        patients.append({'data': df, 'patient_id': patient_id, 'min_time': min_time, 'max_time': max_time})
        if global_min is None and global_max is None:
            global_min = min_time
            global_max = max_time
        else:
            if min_time < global_min:
                global_min = min_time
            if max_time > global_max:
                global_max = max_time
    return patients, global_min, global_max

def schedule_trainings(experiment: str, simulator: Simulator, min_time: pd.Timestamp) -> None:
    if experiment == 'RetrainAfterTime':
        for i in range(1, 4):
            train_event = Event(
                time=min_time + pd.DateOffset(years=i),
                priority=1,
                event_type='TRAIN',
                payload={},
            )
            simulator.schedule_event(train_event)
            test_event = Event(
                time=min_time + pd.DateOffset(years=(i+1)) - pd.DateOffset(days=1),
                priority=2,
                event_type='INFERENCE',
                payload={'last_training_time': min_time + pd.DateOffset(years=i)},
            )
            simulator.schedule_event(test_event)
    elif experiment == 'RetrainEachNDTsActivated':
        ActivationPatientsMonitor(
            simulator=simulator,
            activation_threshold=50,
        )
    elif experiment == 'RetrainAfterPerformanceDrift':
        PerformanceDriftMonitor(
            simulator=simulator,
            bootstrap_months=9,
            inference_interval_days=1,
            retraining_delay_days=1,
            metric_name='rmse',
            degradation_threshold=0.20,
            degraded_dt_fraction_threshold=0.30,
            min_comparable_dts=5,
            threshold_mode='relative',
            higher_is_worse=True,
        )

#@track_emissions
def run_simulation(seed: int, experiment: str) -> None:
    seed_everything(seed)
    all_patients, min_time, max_time = load_patients(data_folder)

    print(f'Found {len(all_patients)} patients')
    print(f'Min: {min_time}, Max: {max_time}')

    simulator = Simulator(data_folder, experiment, min_time, max_time, config, seed)

    # Schedule patients activation and deactivation
    for patient in all_patients:
        event_active = Event(
            time=patient['min_time'],
            priority=0,
            event_type='PATIENT_BECOMES_ACTIVE',
            payload=patient,
        )

        event_inactive = Event(
            time=patient['max_time'],
            priority=0,
            event_type='PATIENT_BECOMES_INACTIVE',
            payload=patient,
        )
        simulator.schedule_event(event_active)
        simulator.schedule_event(event_inactive)

    # Schedule trainings and inferences
    schedule_trainings(experiment, simulator, min_time)

    simulator.start()

if __name__ == "__main__":

    config = LearningConfig()
    data_folder = 'T1DiabetesGranada/split'
    seeds = [0]
    experiments = ['RetrainEachNDTsActivated', 'RetrainAfterTime'] # Add 'RetrainAfterPerformanceDrift' to enable drift-driven retraining

    for experiment in experiments:
        Path(f'{config.data_export_path}/{experiment}').mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_simulation(seed, experiment)
