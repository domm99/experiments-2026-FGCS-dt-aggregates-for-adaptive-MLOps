import torch
import pandas as pd
from torch import nn
from src.distributed.DT import DT
from src.distributed.LearningConfig import LearningConfig
from src.distributed.utils import (
    GlucoseClassifierLSTM,
    compute_train_stats,
    normalize_series,
    create_train_val_loaders,
    evaluate,
)

class DTAggregate:

    def __init__(self, config: LearningConfig, experiment: str, seed: int):
        self._model = GlucoseClassifierLSTM(
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
        self._experiment = experiment

    def update_data_from_dts(self, current_time: pd.Timestamp) -> None:
        self._dts_data = {}
        for dt_id, dt in self._active_dts.items():
            try:
                patient_series = dt.get_data(current_time)
                if patient_series is None:
                    print(f'Skipping DT {dt_id} during training: not enough history yet')
                    continue
                self._dts_data[dt_id] = patient_series
            except Exception as exc:
                print(f'Skipping DT {dt_id} during training: {exc}')

    def register_active_dt(self, local_dt: DT, patient_id: str) -> None:
        self._active_dts[patient_id] = local_dt

    def unregister_active_dt(self, patient_id: str) -> None:
        self._active_dts.pop(patient_id, None)

    @property
    def active_dts(self) -> list[DT]:
        return list(self._active_dts.values())

    @property
    def trainable_dt_count(self) -> int:
        return sum(1 for series in self._dts_data.values() if series is not None)

    @property
    def model(self) -> dict[str, torch.Tensor]:
        return self._model.state_dict()

    @property
    def statistics(self) -> tuple[float, float]:
        return self._last_mean, self._last_std

    def notify_new_model(self):
        for dt in self._active_dts.values():
            dt.model = (self._model.state_dict(), self._last_mean, self._last_std)

    def train(self, current_time: pd.Timestamp) -> None:
        self._model = GlucoseClassifierLSTM(
            hidden_size = self._config.hidden_size,
            num_layers = self._config.layers,
            dropout = self._config.dropout,
        ).to(self._config.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.learning_rate)
        history: list[dict[str, float]] = []
        patients_series_raw = [series for series in self._dts_data.values() if series is not None]
        if not patients_series_raw:
            print('Skipping training: no DT has enough history for training windows yet')
            return False
        mean, std = compute_train_stats(patients_series_raw)
        normalized_series = normalize_series(patients_series_raw, mean, std)
        train_loader, val_loader = create_train_val_loaders(
            patient_series = normalized_series,
            sequence_length = self._config.sequence_length,
            stride = self._config.stride,
            batch_size = self._config.batch_size,
        )

        for epoch in range(1, self._config.epochs + 1):
            self._model.train()
            train_loss_sum = 0.0
            train_correct = 0
            total_samples = 0
            for x, y in train_loader:
                x = x.to(self._device)
                y = y.to(self._device)

                optimizer.zero_grad()
                logits = self._model(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

                batch_size = y.size(0)
                train_loss_sum += loss.item() * batch_size
                train_correct += (logits.argmax(dim=1) == y).sum().item()
                total_samples += batch_size

            val_metrics = evaluate(self._model, val_loader, self._device)

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(total_samples, 1),
                "train_accuracy": train_correct / max(total_samples, 1),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(epoch_log)
            print(
                f"Epoch {epoch:02d} | "
                f"train_acc={epoch_log['train_accuracy']:.4f} | "
                f"val_acc={epoch_log['val_accuracy']:.4f} | "
                f"val_loss={epoch_log['val_loss']:.4f}"
            )

        metrics_df = pd.DataFrame(history)
        metrics_df.to_csv(f'{self._config.data_export_path}/{self._experiment}/training_{current_time}-seed_{self._seed}.csv', index=False)
        self._last_mean = mean
        self._last_std = std
