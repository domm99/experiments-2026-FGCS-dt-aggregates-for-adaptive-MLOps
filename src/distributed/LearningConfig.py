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
    epochs: int = 12
    stride: int = 12
    dropout: float = 0.0
    hidden_size: int = 64
    batch_size: int = 256
    val_ratio: float = 0.2
    train_ratio: float = 0.8
    sequence_length: int = 12
    learning_rate: float = 1e-3
    prediction_horizon: int = 3
    data_export_path: str = 'data'
    device: torch.device = get_current_device()