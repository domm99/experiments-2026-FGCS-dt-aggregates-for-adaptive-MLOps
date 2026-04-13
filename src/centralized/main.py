from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
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
        patient_series: list[PatientSeries],
        sequence_length: int,
        split: str,
        stride: int,
    ) -> None:
        self.patient_series = patient_series
        self.sequence_length = sequence_length
        self.samples: list[tuple[int, int]] = []

        for patient_index, series in enumerate(patient_series):
            if split == "train":
                start_target = sequence_length
                end_target = series.train_end
            elif split == "val":
                start_target = max(sequence_length, series.train_end)
                end_target = series.val_end
            elif split == "test":
                start_target = max(sequence_length, series.val_end)
                end_target = len(series.values)
            else:
                raise ValueError(f"Unsupported split: {split}")

            for target_idx in range(start_target, end_target, stride):
                self.samples.append((patient_index, target_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_index, target_idx = self.samples[index]
        series = self.patient_series[patient_index].values
        x = series[target_idx - self.sequence_length : target_idx].unsqueeze(-1)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Centralized glucose forecasting with a simple LSTM.")
    parser.add_argument("--data-dir", type=Path, default=Path("T1DiabetesGranada/split"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/centralized"))
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-plot-series", type=int, default=5)
    parser.add_argument("--plot-windows-per-series", type=int, default=4)
    parser.add_argument("--plot-window-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_patient_series(
    data_dir: Path,
    sequence_length: int,
    train_ratio: float,
    val_ratio: float,
) -> list[PatientSeries]:
    patient_series: list[PatientSeries] = []

    for csv_path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(
            df["Measurement_date"] + " " + df["Measurement_time"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        df = df.dropna(subset=["timestamp", "Measurement"]).sort_values("timestamp")
        if len(df) <= sequence_length + 2:
            continue

        values = torch.tensor(df["Measurement"].astype(float).to_numpy(), dtype=torch.float32)
        n_total = len(values)
        train_end = max(sequence_length + 1, int(n_total * train_ratio))
        val_end = max(train_end + 1, int(n_total * (train_ratio + val_ratio)))
        val_end = min(val_end, n_total - 1)

        if val_end <= train_end or n_total - val_end < 1:
            continue

        patient_series.append(
            PatientSeries(
                patient_id=str(df["Patient_ID"].iloc[0]),
                timestamps=df["timestamp"].tolist(),
                values=values,
                train_end=train_end,
                val_end=val_end,
            )
        )

    if not patient_series:
        raise RuntimeError(f"No valid patient series found in {data_dir}")

    return patient_series


def compute_train_stats(patient_series: list[PatientSeries]) -> tuple[float, float]:
    train_values = torch.cat([series.values[: series.train_end] for series in patient_series])
    mean = train_values.mean().item()
    std = train_values.std(unbiased=False).item()
    return mean, std if std > 0 else 1.0


def normalize_series(patient_series: list[PatientSeries], mean: float, std: float) -> list[PatientSeries]:
    normalized: list[PatientSeries] = []
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


def create_loaders(
    patient_series: list[PatientSeries],
    sequence_length: int,
    stride: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = GlucoseWindowDataset(patient_series, sequence_length, split="train", stride=stride)
    val_dataset = GlucoseWindowDataset(patient_series, sequence_length, split="val", stride=stride)
    test_dataset = GlucoseWindowDataset(patient_series, sequence_length, split="test", stride=stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    mean: float,
    std: float,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_abs_error_sum = 0.0
        train_sq_error_sum = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = nn.functional.mse_loss(preds, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            train_loss_sum += loss.item() * batch_size
            preds_denorm = preds.detach().cpu() * std + mean
            y_denorm = y.detach().cpu() * std + mean
            train_abs_error_sum += torch.abs(preds_denorm - y_denorm).sum().item()
            train_sq_error_sum += ((preds_denorm - y_denorm) ** 2).sum().item()
            total_samples += batch_size

        val_metrics = evaluate(model, val_loader, device, mean, std)

        epoch_log = {
            "epoch": epoch,
            "train_mse": train_loss_sum / max(total_samples, 1),
            "train_mae": train_abs_error_sum / max(total_samples, 1),
            "train_rmse": math.sqrt(train_sq_error_sum / max(total_samples, 1)),
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
        }
        history.append(epoch_log)
        print(
            f"Epoch {epoch:02d} | "
            f"train_rmse={epoch_log['train_rmse']:.4f} | "
            f"val_rmse={epoch_log['val_rmse']:.4f} | "
            f"val_mae={epoch_log['val_mae']:.4f}"
        )

    return history


def predict_patient(
    model: nn.Module,
    series: PatientSeries,
    sequence_length: int,
    start_idx: int,
    end_idx: int,
    device: torch.device,
    mean: float,
    std: float,
) -> dict[str, list[float] | list[str] | str]:
    windows = []
    actuals = []
    timestamps = []

    for target_idx in range(max(sequence_length, start_idx), end_idx):
        x = series.values[target_idx - sequence_length : target_idx].unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            pred = model(x).cpu().item()
        windows.append(pred * std + mean)
        actuals.append(series.values[target_idx].item() * std + mean)
        timestamps.append(series.timestamps[target_idx].isoformat())

    return {
        "patient_id": series.patient_id,
        "timestamps": timestamps,
        "actual": actuals,
        "predicted": windows,
    }


def collect_test_series_predictions(
    model: nn.Module,
    patient_series: list[PatientSeries],
    sequence_length: int,
    device: torch.device,
    mean: float,
    std: float,
    num_plot_series: int,
) -> list[dict[str, list[float] | list[str] | str]]:
    selected = sorted(patient_series, key=lambda item: len(item.values), reverse=True)[:num_plot_series]
    predictions = []
    for series in selected:
        test_length = len(series.values) - series.val_end
        if test_length <= 0:
            continue
        predictions.append(
            {
                **predict_patient(
                    model=model,
                    series=series,
                    sequence_length=sequence_length,
                    start_idx=series.val_end,
                    end_idx=len(series.values),
                    device=device,
                    mean=mean,
                    std=std,
                ),
                "test_start_idx": series.val_end,
            }
        )
    return predictions


def save_loss_plot(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [entry["train_rmse"] for entry in history], label="Train RMSE", marker="o", linewidth=2)
    plt.plot(epochs, [entry["val_rmse"] for entry in history], label="Validation RMSE", marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (mg/dL)")
    plt.title("Training and validation error over time")
    if len(epochs) == 1:
        plt.xlim(0.5, 1.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def save_prediction_plots(
    predictions: list[dict[str, list[float] | list[str] | str]],
    output_dir: Path,
    windows_per_series: int,
    window_size: int,
) -> None:
    for item in predictions:
        timestamps = pd.to_datetime(item["timestamps"])
        actual = item["actual"]
        predicted = item["predicted"]
        patient_id = item["patient_id"]
        total_points = len(actual)
        if total_points == 0:
            continue

        effective_window = min(window_size, total_points)
        if total_points <= effective_window:
            start_positions = [0]
        else:
            max_start = total_points - effective_window
            if windows_per_series <= 1:
                start_positions = [max_start // 2]
            else:
                start_positions = sorted(
                    {
                        int(round(index * max_start / (windows_per_series - 1)))
                        for index in range(windows_per_series)
                    }
                )

        for window_index, start_idx in enumerate(start_positions, start=1):
            end_idx = min(start_idx + effective_window, total_points)
            window_timestamps = timestamps[start_idx:end_idx]
            window_actual = actual[start_idx:end_idx]
            window_predicted = predicted[start_idx:end_idx]

            plt.figure(figsize=(12, 4.5))
            plt.plot(window_timestamps, window_actual, label="Actual", linewidth=1.8)
            plt.plot(window_timestamps, window_predicted, label="Predicted", linewidth=1.5, alpha=0.9)
            plt.xlabel("Time")
            plt.ylabel("Glucose (mg/dL)")
            plt.title(
                f"Test forecast for patient {patient_id} "
                f"(window {window_index}/{len(start_positions)})"
            )
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"prediction_{patient_id}_window_{window_index}.png", dpi=150)
            plt.close()


def save_summary(
    args: argparse.Namespace,
    history: list[dict[str, float]],
    test_metrics: dict[str, float],
    mean: float,
    std: float,
    patient_series: list[PatientSeries],
    output_dir: Path,
) -> None:
    summary = {
        "config": vars(args),
        "train_stats": {"mean": mean, "std": std},
        "num_patients": len(patient_series),
        "history": history,
        "test_metrics": test_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_series_raw = load_patient_series(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    mean, std = compute_train_stats(patient_series_raw)
    patient_series = normalize_series(patient_series_raw, mean, std)
    train_loader, val_loader, test_loader = create_loaders(
        patient_series=patient_series,
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
    )

    model = ForecastLSTM(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(
        f"Loaded {len(patient_series)} patients | "
        f"train_windows={len(train_loader.dataset)} | "
        f"val_windows={len(val_loader.dataset)} | "
        f"test_windows={len(test_loader.dataset)} | "
        f"device={device}"
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mean=mean,
        std=std,
    )

    test_metrics = evaluate(model, test_loader, device, mean, std)
    print(
        f"Test metrics | mse={test_metrics['mse']:.4f} | "
        f"rmse={test_metrics['rmse']:.4f} | mae={test_metrics['mae']:.4f}"
    )

    predictions = collect_test_series_predictions(
        model=model,
        patient_series=patient_series,
        sequence_length=args.sequence_length,
        device=device,
        mean=mean,
        std=std,
        num_plot_series=args.num_plot_series,
    )
    save_loss_plot(history, args.output_dir)
    save_prediction_plots(
        predictions,
        args.output_dir,
        windows_per_series=args.plot_windows_per_series,
        window_size=args.plot_window_size,
    )
    save_summary(args, history, test_metrics, mean, std, patient_series, args.output_dir)
    torch.save(model.state_dict(), args.output_dir / "model.pt")


if __name__ == "__main__":
    main()
