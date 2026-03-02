import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.metrics import metric, print_metrics
from utils.preprocessing import RevIN

# ==========================================
# 1. Moving Average Block (Decomposition)
# ==========================================
class Series_Decomp(nn.Module):
    """
    Separates a time series into Trend and Seasonal components.
    """
    def __init__(self, kernel_size):
        super(Series_Decomp, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x shape: [Batch*Variates, Seq_Len]
        # Padding to maintain sequence length
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # Trend is the smooth part (Moving Average)
        trend = self.avg(x_padded.unsqueeze(1)).squeeze(1)
        # Seasonal is the "wiggles" (Residual after removing trend)
        seasonal = x - trend
        return seasonal, trend

# ==========================================
# 2. Main QDLinear Model
# ==========================================
class QuLTSF_Decomp_Model(nn.Module):
    def __init__(self, configs):
        super(QuLTSF_Decomp_Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_qubits = configs.num_qubits
        self.num_layers = configs.num_layers
        self.QML_device = configs.QML_device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. RevIN for data stabilization
        num_features = getattr(configs, 'num_features', 21)
        self.revin_layer = RevIN(num_features)

        # 2. Decomposition Block (Kernel size 25 covers ~4 hours of weather data)
        self.decomposition = Series_Decomp(kernel_size=25)

        # 3. BRANCH 1: Classical Trend Model
        self.trend_model = nn.Linear(self.seq_len, self.pred_len)

        # 4. BRANCH 2: Quantum Seasonal Model
        self.seasonal_model = Seasonal_Quantum_Engine(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
            QML_device=self.QML_device
        )

    def forward(self, x):
        # A. Normalize
        x = self.revin_layer(x, 'norm')
        
        # B. Prep for Channel Independence
        B, L, M = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * M, L)
        
        # C. Decompose
        seasonal_init, trend_init = self.decomposition(x_flat)
        
        # D. Process Branch 1 (Trend)
        trend_output = self.trend_model(trend_init)
        
        # E. Process Branch 2 (Seasonal - Quantum)
        seasonal_output = self.seasonal_model(seasonal_init)
        
        # F. Re-Combine
        out = seasonal_output + trend_output
        
        # G. Reshape & Denormalize
        out = out.reshape(B, M, -1).permute(0, 2, 1)
        out = self.revin_layer(out, 'denorm')
        return out

    def summary(self):
        print("\n" + "="*85)
        print(f"{'QDLinear (Quantum-Decomposition) Model Summary':^85}")
        print("="*85)
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                p_count = param.numel()
                total_params += p_count
                print(f"{name:<50} | {str(list(param.shape)):<20} | {p_count:,}")
        print("-" * 85)
        print(f"Trend Path: Classical Linear (Handles Drift)")
        print(f"Seasonal Path: Quantum VQC (Handles Cycles)")
        print(f"Total Trainable Parameters: {total_params:,}")
        print("="*85 + "\n")

    def train_model(self, train_loader, val_loader):
        self.to(self.device)
        criterion = nn.MSELoss()
        
        # Parameter Splitting
        quantum_params = list(self.seasonal_model.q_layer.parameters())
        quantum_ids = list(map(id, quantum_params))
        classical_params = filter(lambda p: id(p) not in quantum_ids, self.parameters())

        optimizer = torch.optim.Adam([
            {'params': classical_params, 'lr': self.configs.lr},
            {'params': quantum_params, 'lr': self.configs.lr * 0.1}
        ])
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        for epoch in range(self.configs.epochs):
            self.train()
            train_loss = []
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss.append(loss.item())

            self.eval()
            val_loss = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self(batch_x)
                    val_loss.append(criterion(outputs, batch_y).item())
            
            avg_val_loss = np.mean(val_loss)
            scheduler.step(avg_val_loss)
            print(f"Epoch {epoch+1}: Train MSE: {np.mean(train_loss):.5f} | Val MSE: {avg_val_loss:.5f}")

    def test_model(self, test_loader, scaler, plot_idx=0):
        self.eval(); self.to(self.device)
        preds, trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self(batch_x)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print_metrics(mae, mse, rmse, mape, mspe, tag="QDLINEAR TEST")

        # Visuals
        plt.figure(figsize=(10, 5))
        input_real = scaler.inverse_transform(test_loader.dataset.x[0])
        true_real = scaler.inverse_transform(trues[0])
        pred_real = scaler.inverse_transform(preds[0])
        plt.plot(input_real[:, plot_idx], label='History', color='black')
        plt.plot(np.arange(self.seq_len, self.seq_len+self.pred_len), true_real[:, plot_idx], label='True', color='green')
        plt.plot(np.arange(self.seq_len, self.seq_len+self.pred_len), pred_real[:, plot_idx], label='QD-Quantum', color='red', ls='--')
        plt.legend(); plt.title(f"QDLinear Forecast (Variate {plot_idx})"); plt.show()
        return mse, mae

    def save_model(self, scaler, folder="checkpoints", name="qdlinear_model"):
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save({'state_dict': self.state_dict(), 'configs': self.configs}, f"{folder}/{name}.pth")
        joblib.dump(scaler, f"{folder}/{name}_scaler.pkl")

    @classmethod
    def load_model(cls, folder="checkpoints", name="qdlinear_model", device='cpu'):
        checkpoint = torch.load(f"{folder}/{name}.pth", map_location=device)
        model = cls(checkpoint['configs'])
        model.load_state_dict(checkpoint['state_dict'])
        scaler = joblib.load(f"{folder}/{name}_scaler.pkl")
        return model, scaler

# ==========================================
# 3. Seasonal Quantum Engine (Detrended Input)
# ==========================================
class Seasonal_Quantum_Engine(nn.Module):
    def __init__(self, seq_len, pred_len, num_qubits, num_layers, QML_device):
        super().__init__()
        self.dev = qml.device(QML_device, wires=num_qubits)

        @qml.qnode(self.dev, interface='torch', diff_method="best")
        def quantum_function(inputs, weights):
            # Input is detrended (seasonal), ideally centered at 0
            qml.AmplitudeEmbedding(features=inputs, wires=range(num_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.input_linear = nn.Linear(seq_len, 2 ** num_qubits)
        self.q_layer = qml.qnn.TorchLayer(quantum_function, {'weights': (num_layers, num_qubits, 3)})
        self.output_linear = nn.Linear(num_qubits, pred_len)

    def forward(self, x):
        x = self.input_linear(x)
        x = x + 1e-6 # Stability
        x = self.q_layer(x)
        x = self.output_linear(x)
        return x