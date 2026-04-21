import math
import torch
import pandas as pd
from torch import nn
from src.distributed.DT import DT
from collections import OrderedDict
from src.distributed.LearningConfig import LearningConfig
from src.distributed.utils import ForecastLSTM, PatientSeries, compute_train_stats, normalize_series, create_train_val_loaders, evaluate

class DTAggregate:

    def __init__(self, config: LearningConfig, seed: int):
        self._model = ForecastLSTM(
            hidden_size = config.hidden_size,
            num_layers = config.layers,
            dropout = config.dropout,
        )
        self._device = config.device
        self._config = config
        self._dts_data = {}
        self._seed = seed
        self._active_dts = {}
        self._last_mean = 0.0
        self._last_std = 0.0

    def update_data_from_dts(self, current_time: pd.Timestamp) -> None:
        for dt_id, dt in self._active_dts.items():
            self._dts_data[dt_id] = dt.get_data(current_time)

    def register_active_dt(self, local_dt: DT, patient_id: str) -> None:
        self._active_dts[patient_id] = local_dt

    def unregister_active_dt(self, patient_id: str) -> None:
        self._active_dts.pop(patient_id, None)

    @property
    def active_dts(self) -> list[DT]:
        return list(self._active_dts.values())

    @property
    def model(self) -> OrderedDict[str, torch.Tensor]:
        return self._model.state_dict()

    def notify_new_model(self):
        for dt in self._active_dts.values():
            dt.model = (self._model.state_dict(), self._last_mean, self._last_std)

    def train(self, current_time: pd.Timestamp) -> None:
        self._model = ForecastLSTM(
            hidden_size = self._config.hidden_size,
            num_layers = self._config.layers,
            dropout = self._config.dropout,
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.learning_rate)
        history: list[dict[str, float]] = []
        patients_series_raw = list(self._dts_data.values())
        mean, std = compute_train_stats(patients_series_raw)
        normalized_series = normalize_series(patients_series_raw, mean, std)
        train_loader, val_loader = create_train_val_loaders(
            patient_series = normalized_series,
            sequence_length = self._config.sequence_length,
            prediction_horizon = self._config.prediction_horizon,
            stride = self._config.stride,
            batch_size = self._config.batch_size,
        )

        for epoch in range(1, self._config.epochs + 1):
            self._model.train()
            train_loss_sum = 0.0
            train_abs_error_sum = 0.0
            train_sq_error_sum = 0.0
            total_samples = 0
            for x, y in train_loader:
                x = x.to(self._device)
                y = y.to(self._device)

                optimizer.zero_grad()
                preds = self._model(x)
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

            val_metrics = evaluate(self._model, val_loader, self._device, mean, std)

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

        metrics_df = pd.DataFrame(history)
        metrics_df.to_csv(f'{self._config.data_export_path}/training_{current_time}-seed_{self._seed}.csv', index=False)
        self._last_mean = mean
        self._last_std = std

