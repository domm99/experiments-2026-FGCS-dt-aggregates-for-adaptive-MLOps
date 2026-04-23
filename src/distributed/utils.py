import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

@dataclass
class PatientSeries:
    patient_id: str
    timestamps: list[pd.Timestamp]
    values: torch.Tensor
    train_end: int
    val_end: int


class GlucoseWindowDataset(Dataset):

    def __init__(
        self,
        patient_series: list[PatientSeries] | PatientSeries,
        sequence_length: int,
        prediction_horizon: int,
        split: str,
        stride: int,
    ) -> None:
        if isinstance(patient_series, PatientSeries):
            self.patient_series = [patient_series]
        else:
            self.patient_series = patient_series
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.samples: list[tuple[int, int]] = []

        for patient_index, series in enumerate(self.patient_series):
            if split == "train":
                start_input_end = sequence_length
                end_input_end = series.train_end - prediction_horizon
            elif split == "val":
                start_input_end = max(sequence_length, series.train_end - prediction_horizon)
                end_input_end = series.val_end - prediction_horizon
            elif split == "test":
                # At inference time the local DT receives only the interval that
                # must be evaluated, so the whole series is treated as test.
                start_input_end = sequence_length
                end_input_end = len(series.values) - prediction_horizon
            else:
                raise ValueError(f"Unsupported split: {split}")

            if end_input_end <= start_input_end:
                continue

            for input_end_idx in range(start_input_end, end_input_end, stride):
                self.samples.append((patient_index, input_end_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_index, input_end_idx = self.samples[index]
        series = self.patient_series[patient_index].values
        target_idx = input_end_idx + self.prediction_horizon
        x = series[input_end_idx - self.sequence_length : input_end_idx].unsqueeze(-1)
        y = series[target_idx]
        return x, y


class ForecastLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        prediction = self.head(output[:, -1, :]).squeeze(-1)
        return prediction


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _prepare_patient_dataframe(patient_dataframe: pd.DataFrame) -> pd.DataFrame:
    df = patient_dataframe.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["Measurement_date"] + " " + df["Measurement_time"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
    df = df.dropna(subset=["timestamp", "Measurement"]).sort_values("timestamp")
    return df


def load_patient_series(
    patient_id: str,
    patient_dataframe: pd.DataFrame,
    sequence_length: int,
    prediction_horizon: int,
    train_ratio: float,
) -> PatientSeries | None:
    df = _prepare_patient_dataframe(patient_dataframe)

    min_required = sequence_length + prediction_horizon + 1
    if len(df) <= min_required:
        #raise Exception(f"No valid patient series found for {patient_id}")
        return None
    values = torch.tensor(
        df["Measurement"].astype(float).to_numpy(),
        dtype=torch.float32,
    )
    n_total = len(values)

    train_end = max(min_required, int(n_total * train_ratio))

    if train_end >= n_total:
        raise Exception(f"Training split is too large for {patient_id}")

    return PatientSeries(
        patient_id=str(df["Patient_ID"].iloc[0]),
        timestamps=df["timestamp"].tolist(),
        values=values,
        train_end=train_end,
        val_end=n_total,
    )


def load_test_patient_series(
    patient_id: str,
    patient_dataframe: pd.DataFrame,
    sequence_length: int,
    prediction_horizon: int,
) -> PatientSeries | None:
    df = _prepare_patient_dataframe(patient_dataframe)

    min_required = sequence_length + prediction_horizon + 1
    if len(df) < min_required:
        return None

    values = torch.tensor(
        df["Measurement"].astype(float).to_numpy(),
        dtype=torch.float32,
    )

    return PatientSeries(
        patient_id=str(df["Patient_ID"].iloc[0]) if not df.empty else patient_id,
        timestamps=df["timestamp"].tolist(),
        values=values,
        train_end=len(values),
        val_end=len(values),
    )


def compute_train_stats(patient_series: list[PatientSeries]) -> tuple[float, float]:
    train_values = torch.cat([series.values[: series.train_end] for series in patient_series])
    mean = train_values.mean().item()
    std = train_values.std(unbiased=False).item()
    return mean, std if std > 0 else 1.0


def normalize_series(patient_series: list[PatientSeries] | PatientSeries, mean: float, std: float) -> list[PatientSeries]:
    normalized: list[PatientSeries] = []

    if isinstance(patient_series, PatientSeries):
        patient_series = [patient_series]

    for series in patient_series:
        normalized.append(
            PatientSeries(
                patient_id=series.patient_id,
                timestamps=series.timestamps,
                values=(series.values - mean) / std,
                train_end=series.train_end,
                val_end=series.val_end,
            )
        )
    return normalized


def create_train_val_loaders(
    patient_series: list[PatientSeries],
    sequence_length: int,
    prediction_horizon: int,
    stride: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = GlucoseWindowDataset(
        patient_series,
        sequence_length,
        prediction_horizon,
        split="train",
        stride=stride,
    )
    val_dataset = GlucoseWindowDataset(
        patient_series,
        sequence_length,
        prediction_horizon,
        split="val",
        stride=stride,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def create_test_loaders(
    patient_series: list[PatientSeries] | PatientSeries,
    sequence_length: int,
    prediction_horizon: int,
    stride: int,
    batch_size: int,
) -> DataLoader:
    test_dataset = GlucoseWindowDataset(
        patient_series,
        sequence_length,
        prediction_horizon,
        split="test",
        stride=stride,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(predictions - targets))

def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss = nn.functional.mse_loss(preds, y, reduction="sum")

            total_loss += loss.item()
            total_samples += y.size(0)
            predictions_list.append(preds.cpu())
            targets_list.append(y.cpu())

    predictions = torch.cat(predictions_list) * std + mean
    targets = torch.cat(targets_list) * std + mean

    return {
        "mse": total_loss / max(total_samples, 1),
        "mae": mae(predictions, targets).item(),
        "rmse": rmse(predictions, targets).item(),
    }
