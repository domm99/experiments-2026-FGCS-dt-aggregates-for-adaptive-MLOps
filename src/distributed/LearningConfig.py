import torch
from dataclasses import dataclass

@dataclass
class LearningConfig:

    sequence_length: int = 12
    prediction_horizon: int = 3
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    device: torch.device = torch.device("cpu") ## TODO fix this
    stride: int = 1, ## TODO fix
    batch_size: int = 1,  ## TODO fix