import glob
import pandas as pd
from pathlib import Path
from codecarbon import track_emissions
from src.distributed.utils import seed_everything
from src.distributed.Simulator import Simulator, Event
from src.distributed.LearningConfig import LearningConfig
from src.distributed.Monitors import (
    ActivationPatientsMonitor,
    AdwinGlobalErrorMonitor,
    PerformanceDriftMonitor,
    PeriodicInferenceMonitor,
)


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

def schedule_trainings(experiment: str, simulator: Simulator, min_time: pd.Timestamp, max_time: pd.Timestamp) -> None:
    if experiment == 'RetrainAfterTime':
        PeriodicInferenceMonitor(
            simulator=simulator,
            inference_interval_days=simulator.config.drift_inference_interval_days,
        )
        current_time = min_time
        i = 0
        months_step = 3
        while current_time < max_time:
            train_event = Event(
                time=min_time + pd.DateOffset(months=months_step*i),
                priority=1,
                event_type='TRAIN',
                payload={},
            )
            simulator.schedule_event(train_event)
            current_time = current_time + pd.DateOffset(months=months_step)
            i += 1
    elif experiment == 'RetrainEachNDTsActivated':
        ActivationPatientsMonitor(
            simulator=simulator,
            activation_threshold=20,
        )
        PeriodicInferenceMonitor(
            simulator=simulator,
            inference_interval_days=simulator.config.drift_inference_interval_days,
        )
    elif experiment == 'RetrainAfterPerformanceDrift':
        PerformanceDriftMonitor(
            simulator=simulator,
            bootstrap_months=config.drift_bootstrap_months,
            inference_interval_days=config.drift_inference_interval_days,
            retraining_delay_days=config.drift_retraining_delay_days,
            metric_name=config.drift_metric_name,
            degradation_threshold=config.drift_degradation_threshold,
            degraded_dt_fraction_threshold=config.degraded_dt_fraction_threshold,
            metric_floor=config.drift_metric_floor,
            min_comparable_dts=config.drift_min_comparable_dts,
            threshold_mode=config.drift_threshold_mode,
            higher_is_worse=config.drift_higher_is_worse,
        )
    elif experiment == 'adwin_global_error':
        AdwinGlobalErrorMonitor(
            simulator=simulator,
            bootstrap_months=simulator.config.drift_bootstrap_months,
            inference_interval_days=simulator.config.drift_inference_interval_days,
            delta=simulator.config.adwin_delta,
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
    schedule_trainings(experiment, simulator, min_time, max_time)

    simulator.start()

if __name__ == "__main__":

    config = LearningConfig()
    data_folder = 'T1DiabetesGranada/split-labeled'
    seeds = [0]
    experiments = ['adwin_global_error'] #['adwin_global_error', 'RetrainEachNDTsActivated', 'RetrainAfterTime', 'RetrainAfterPerformanceDrift']

    for experiment in experiments:
        Path(f'{config.data_export_path}/{experiment}').mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_simulation(seed, experiment)
