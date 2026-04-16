import torch
from src.common import ForecastLSTM

class DTAggregate:

    def __init__(self, device: torch.device):
        self._model = None
        self.epochs = 10
        self.learning_rate = 0.001 ## TODO fix this
        self._data = None ## TODO set this
        self.device = device


    def notify_retraining_needed(self):
        pass

    def notify_new_data(self):
        pass

    def train(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self._model = ForecastLSTM(hidden_size = 64, num_layers = 1, dropout = 0.0)
        history: list[dict[str, float]] = []
        train_loader = ... ## TODO define this

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            train_loss_sum = 0.0
            train_abs_error_sum = 0.0
            train_sq_error_sum = 0.0
            total_samples = 0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                preds = self._model(x) ## TODO check this warning
                loss = nn.functional.mse_loss(preds, y)
                loss.backward()
                optimizer.step()

                batch_size = y.size(0)
                train_loss_sum += loss.item() * batch_size
                preds_denorm = preds.detach().cpu() * std + mean ## TODO Understand what the fuck mean and std are
                y_denorm = y.detach().cpu() * std + mean
                train_abs_error_sum += torch.abs(preds_denorm - y_denorm).sum().item()
                train_sq_error_sum += ((preds_denorm - y_denorm) ** 2).sum().item()
                total_samples += batch_size

            val_metrics = evaluate(model, val_loader, device, mean, std)  ## TODO add evaluate

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
