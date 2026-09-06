import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.metrics import metric, print_metrics
from utils.preprocessing import RevIN


class StiefelUnitaryLayer(nn.Module):
    """Trainable complex unitary matrix updated on the Stiefel manifold."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim, dtype=torch.complex64))

    def forward(self, x):
        return torch.matmul(x, self.weight)


def apply_cayley_updates(model, lr, neumann_terms=3):
    """Apply the notebook's Neumann-approximated Cayley update."""
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, StiefelUnitaryLayer):
                continue

            weight = module.weight
            grad = weight.grad
            if grad is None:
                continue

            weight_h = weight.resolve_conj().transpose(-2, -1)
            grad_h = grad.resolve_conj().transpose(-2, -1)
            skew_hermitian = grad @ weight_h - weight @ grad_h
            x = (lr / 2.0) * skew_hermitian
            identity = torch.eye(module.dim, device=weight.device, dtype=weight.dtype)

            inverse_approx = identity.clone()
            power = identity
            for _ in range(neumann_terms):
                power = power @ (-x)
                inverse_approx = inverse_approx + power

            weight.copy_(inverse_approx @ (identity - x) @ weight)
            grad.zero_()


class Stiefel_QuLTSF_Model(nn.Module):
    """Long-term forecaster using a stack of learned unitary matrices."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_qubits = configs.num_qubits
        self.num_layers = configs.num_layers
        self.num_features = getattr(configs, 'num_features', 21)
        self.stiefel_lr = getattr(configs, 'stiefel_lr', 0.01)
        self.neumann_terms = getattr(configs, 'neumann_terms', 3)
        self.dim = 2 ** self.num_qubits
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.revin_layer = RevIN(self.num_features)
        self.input_projection = nn.Linear(self.seq_len, self.dim)
        self.unitary_stack = nn.ModuleList(
            StiefelUnitaryLayer(self.dim) for _ in range(self.num_layers)
        )
        self.output_head = nn.Linear(self.dim, self.pred_len)

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1).reshape(batch_size * num_features, seq_len)

        x = self.input_projection(x).to(torch.complex64)
        x = x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-8)
        for layer in self.unitary_stack:
            x = layer(x)

        x = torch.real(x * x.conj())
        x = self.output_head(x)
        x = x.reshape(batch_size, num_features, self.pred_len).permute(0, 2, 1)
        return self.revin_layer(x, 'denorm')

    def summary(self):
        print("\n" + "=" * 85)
        print(f"{'Stiefel-QuLTSF Model Summary':^85}")
        print("=" * 85)
        print(f"State dimension: 2^{self.num_qubits} = {self.dim}")
        print(f"Unitary layers: {self.num_layers} | Neumann terms: {self.neumann_terms}")
        print("-" * 85)
        total_params = 0
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                count = parameter.numel()
                total_params += count
                print(f"{name:<50} | {str(list(parameter.shape)):<20} | {count:,}")
        print("-" * 85)
        print(f"Total Trainable Parameters: {total_params:,}")
        print("=" * 85 + "\n")

    def _classical_parameters(self):
        return list(self.input_projection.parameters()) + \
            list(self.output_head.parameters()) + \
            list(self.revin_layer.parameters())

    def train_model(self, train_loader, val_loader):
        self.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._classical_parameters(), lr=self.configs.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        for epoch in range(self.configs.epochs):
            self.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                for layer in self.unitary_stack:
                    layer.weight.grad = None

                output = self(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._classical_parameters(), max_norm=1.0)
                optimizer.step()
                apply_cayley_updates(self, self.stiefel_lr, self.neumann_terms)
                train_losses.append(loss.item())

            self.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    val_losses.append(criterion(self(batch_x), batch_y).item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)
            print(
                f"Epoch {epoch + 1}: Train MSE: {train_loss:.5f} | "
                f"Val MSE: {val_loss:.5f}"
            )

    def test_model(self, test_loader, scaler, plot_idx=0):
        self.eval()
        self.to(self.device)
        predictions, targets = [], []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = self(batch_x.to(self.device))
                predictions.append(output.cpu().numpy())
                targets.append(batch_y.numpy())

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        mae, mse, rmse, mape, mspe = metric(predictions, targets)
        print_metrics(mae, mse, rmse, mape, mspe, tag="STIEFEL TEST")

        history = scaler.inverse_transform(test_loader.dataset.x[0].numpy())
        target = scaler.inverse_transform(targets[0])
        prediction = scaler.inverse_transform(predictions[0])
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(self.seq_len), history[:, plot_idx], label='History', color='black')
        forecast_axis = np.arange(self.seq_len, self.seq_len + self.pred_len)
        plt.plot(forecast_axis, target[:, plot_idx], label='Ground Truth', color='green')
        plt.plot(forecast_axis, prediction[:, plot_idx], label='Prediction', color='red', linestyle='--')
        plt.legend()
        plt.title(f"Stiefel-QuLTSF Forecast (Variate {plot_idx})")
        plt.show()
        return mse, mae

    def save_model(self, scaler, folder="checkpoints", name="stiefel_qultsf"):
        os.makedirs(folder, exist_ok=True)
        torch.save(
            {'state_dict': self.state_dict(), 'configs': self.configs},
            os.path.join(folder, f"{name}.pth"),
        )
        joblib.dump(scaler, os.path.join(folder, f"{name}_scaler.pkl"))
        print(f"Model and scaler saved to {folder}")

    @classmethod
    def load_model(cls, folder="checkpoints", name="stiefel_qultsf", device='cpu'):
        checkpoint = torch.load(
            os.path.join(folder, f"{name}.pth"),
            map_location=device,
            weights_only=False,
        )
        model = cls(checkpoint['configs'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        scaler = joblib.load(os.path.join(folder, f"{name}_scaler.pkl"))
        return model, scaler
