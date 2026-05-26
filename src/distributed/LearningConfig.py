import torch
from dataclasses import dataclass

def get_current_device():
    device: str = 'cpu'
    if torch.accelerator.is_available():
        current_accelerator = torch.accelerator.current_accelerator()
        if current_accelerator is not None:
            device = current_accelerator.type
    return torch.device(device)


@dataclass
class LearningConfig:
    layers: int = 1
    epochs: int = 4
    stride: int = 12
    dropout: float = 0.0
    hidden_size: int = 64
    batch_size: int = 256
    val_ratio: float = 0.2
    train_ratio: float = 0.8
    sequence_length: int = 12
    learning_rate: float = 1e-3
    label_column: str = 'target_any_within_t'
    drift_bootstrap_months: int = 3
    drift_inference_interval_days: int = 3
    drift_retraining_delay_days: int = 1
    drift_metric_name: str = 'f1_score'
    drift_degradation_threshold: float = 0.1
    drift_metric_floor: float | None = None
    drift_min_comparable_dts: int = 5
    drift_threshold_mode: str = 'relative'
    drift_higher_is_worse: bool = False
    degraded_dt_fraction_threshold: float = 0.2
    adwin_delta: float = 0.002
    data_export_path: str = 'data'
    device: torch.device = get_current_device()
