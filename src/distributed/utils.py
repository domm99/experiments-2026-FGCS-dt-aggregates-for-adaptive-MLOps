import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ("hypo", "normal", "hyper")
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
DEFAULT_LABEL_COLUMN = "target_any_within_t"


@dataclass
class PatientSeries:
    patient_id: str
    timestamps: list[pd.Timestamp]
    values: torch.Tensor
    labels: torch.Tensor
    train_end: int
    val_end: int


class GlucoseWindowDataset(Dataset):

    def __init__(
        self,
        patient_series: list[PatientSeries] | PatientSeries,
        sequence_length: int,
        split: str,
        stride: int,
    ) -> None:
        if isinstance(patient_series, PatientSeries):
            self.patient_series = [patient_series]
        else:
            self.patient_series = patient_series
        self.sequence_length = sequence_length
        self.samples: list[tuple[int, int]] = []

        for patient_index, series in enumerate(self.patient_series):
            if split == "train":
                start_input_end = sequence_length
                end_input_end = series.train_end + 1
            elif split == "val":
                start_input_end = max(sequence_length, series.train_end + 1)
                end_input_end = series.val_end + 1
            elif split == "test":
                # At inference time the local DT receives only the interval that
                # must be evaluated, so the whole series is treated as test.
                start_input_end = sequence_length
                end_input_end = len(series.values) + 1
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
        series = self.patient_series[patient_index]
        target_idx = input_end_idx - 1
        x = series.values[input_end_idx - self.sequence_length : input_end_idx].unsqueeze(-1)
        y = series.labels[target_idx]
        return x, y


class GlucoseClassifierLSTM(nn.Module):
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
        self.head = nn.Linear(hidden_size, len(CLASS_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def encode_labels(labels: pd.Series) -> torch.Tensor:
    normalized = labels.astype(str).str.strip().str.lower()
    invalid_labels = sorted(set(normalized) - set(CLASS_TO_INDEX))
    if invalid_labels:
        raise ValueError(f"Unsupported labels found: {invalid_labels}")
    return torch.tensor(
        normalized.map(CLASS_TO_INDEX).to_numpy(),
        dtype=torch.long,
    )


def decode_label_indices(indices: torch.Tensor) -> list[str]:
    return [CLASS_NAMES[int(index)] for index in indices.tolist()]


def _prepare_patient_dataframe(
    patient_dataframe: pd.DataFrame,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    df = patient_dataframe.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["Measurement_date"] + " " + df["Measurement_time"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
    df = df.dropna(subset=["timestamp", "Measurement", label_column]).sort_values("timestamp")
    return df


def load_patient_series(
    patient_id: str,
    patient_dataframe: pd.DataFrame,
    sequence_length: int,
    train_ratio: float,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> PatientSeries | None:
    df = _prepare_patient_dataframe(patient_dataframe, label_column=label_column)

    min_required = sequence_length + 2
    if len(df) < min_required:
        return None

    values = torch.tensor(
        df["Measurement"].astype(float).to_numpy(),
        dtype=torch.float32,
    )
    labels = encode_labels(df[label_column])
    n_total = len(values)
    train_end = max(sequence_length + 1, int(n_total * train_ratio))

    if train_end >= n_total:
        raise Exception(f"Training split is too large for {patient_id}")

    return PatientSeries(
        patient_id=str(df["Patient_ID"].iloc[0]),
        timestamps=df["timestamp"].tolist(),
        values=values,
        labels=labels,
        train_end=train_end,
        val_end=n_total,
    )


def load_test_patient_series(
    patient_id: str,
    patient_dataframe: pd.DataFrame,
    sequence_length: int,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> PatientSeries | None:
    df = _prepare_patient_dataframe(patient_dataframe, label_column=label_column)

    min_required = sequence_length + 1
    if len(df) < min_required:
        return None

    values = torch.tensor(
        df["Measurement"].astype(float).to_numpy(),
        dtype=torch.float32,
    )
    labels = encode_labels(df[label_column])

    return PatientSeries(
        patient_id=str(df["Patient_ID"].iloc[0]) if not df.empty else patient_id,
        timestamps=df["timestamp"].tolist(),
        values=values,
        labels=labels,
        train_end=len(values),
        val_end=len(values),
    )


def compute_train_stats(patient_series: list[PatientSeries]) -> tuple[float, float]:
    train_values = torch.cat([series.values[: series.train_end] for series in patient_series])
    mean = train_values.mean().item()
    std = train_values.std(unbiased=False).item()
    return mean, std if std > 0 else 1.0


def normalize_series(
    patient_series: list[PatientSeries] | PatientSeries,
    mean: float,
    std: float,
) -> list[PatientSeries]:
    normalized: list[PatientSeries] = []

    if isinstance(patient_series, PatientSeries):
        patient_series = [patient_series]

    for series in patient_series:
        normalized.append(
            PatientSeries(
                patient_id=series.patient_id,
                timestamps=series.timestamps,
                values=(series.values - mean) / std,
                labels=series.labels,
                train_end=series.train_end,
                val_end=series.val_end,
            )
        )
    return normalized


def create_train_val_loaders(
    patient_series: list[PatientSeries],
    sequence_length: int,
    stride: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = GlucoseWindowDataset(
        patient_series,
        sequence_length,
        split="train",
        stride=stride,
    )
    val_dataset = GlucoseWindowDataset(
        patient_series,
        sequence_length,
        split="val",
        stride=stride,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def create_test_loaders(
    patient_series: list[PatientSeries] | PatientSeries,
    sequence_length: int,
    stride: int,
    batch_size: int,
) -> DataLoader:
    test_dataset = GlucoseWindowDataset(
        patient_series,
        sequence_length,
        split="test",
        stride=stride,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def compute_class_weights(
    patient_series: list[PatientSeries] | PatientSeries,
    sequence_length: int,
    stride: int,
    split: str = "train",
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = GlucoseWindowDataset(
        patient_series,
        sequence_length,
        split=split,
        stride=stride,
    )
    class_counts = torch.zeros(len(CLASS_NAMES), dtype=torch.long)

    for patient_index, input_end_idx in dataset.samples:
        series = dataset.patient_series[patient_index]
        target_idx = input_end_idx - 1
        class_index = int(series.labels[target_idx].item())
        class_counts[class_index] += 1

    if class_counts.sum().item() == 0:
        raise RuntimeError(f"No samples available to compute class weights for split={split!r}")

    class_weights = class_counts.sum().float() / (len(CLASS_NAMES) * class_counts.clamp_min(1).float())
    return class_weights, class_counts


def cross_entropy_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float]:
    per_sample_loss = nn.functional.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
    )
    loss_sum = per_sample_loss.sum()

    if class_weights is None:
        normalization = float(targets.numel())
    else:
        normalization = float(class_weights[targets].sum().item())

    normalization = max(normalization, 1.0)
    loss = loss_sum / normalization
    return loss, float(loss_sum.item()), normalization


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_loss_normalization = 0.0
    total_samples = 0
    total_correct = 0
    loss_weights = class_weights.to(device) if class_weights is not None else None

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            predictions = logits.argmax(dim=1)
            _, loss_sum, loss_normalization = cross_entropy_batch(logits, y, loss_weights)

            total_loss += loss_sum
            total_loss_normalization += loss_normalization
            total_samples += y.size(0)
            total_correct += (predictions == y).sum().item()

    return {
        "loss": total_loss / max(total_loss_normalization, 1.0),
        "accuracy": total_correct / max(total_samples, 1),
    }
