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

class QuLTSF_Model(nn.Module):
    def __init__(self, configs):
        super(QuLTSF_Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_qubits = configs.num_qubits
        self.num_layers = configs.num_layers
        self.QML_device = configs.QML_device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Reversible Instance Normalization (RevIN)
        # Assuming 21 variates for weather, but can be dynamic
        num_features = getattr(configs, 'num_features', 21)
        self.revin_layer = RevIN(num_features)

        # 2. Hybrid QML Engine
        self.hybrid_qml_model = Hybrid_QML_Engine(
            lookback_window_size=self.seq_len,
            forecast_window_size=self.pred_len,
            num_qubits=self.num_qubits,
            QML_device=self.QML_device,
            num_layers=self.num_layers
        )

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Variates]
        """
        # A. Instance Normalization
        x = self.revin_layer(x, 'norm')
        
        # B. Quantum-Classical Hybrid Processing
        x = self.hybrid_qml_model(x)
        
        # C. Denormalization
        x = self.revin_layer(x, 'denorm')
        return x

    def summary(self):
        """Prints a summary of the model architecture and parameters."""
        print("\n" + "="*80)
        print(f"{'QuLTSF Model Architecture Summary':^80}")
        print("="*80)
        print(f"{'Layer / Parameter Name':<45} | {'Shape':<20} | {'Params'}")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                p_count = param.numel()
                total_params += p_count
                print(f"{name:<45} | {str(list(param.shape)):<20} | {p_count:,}")
        print("-" * 80)
        print(f"Total Trainable Parameters: {total_params:,}")
        print("="*80 + "\n")

    def train_model(self, train_loader, val_loader):
        self.to(self.device)
        criterion = nn.MSELoss()
        
        # Separate Learning Rates: Classical (Higher) vs Quantum (Lower)
        quantum_params = list(self.hybrid_qml_model.hidden_quantum_layer.parameters())
        quantum_ids = list(map(id, quantum_params))
        classical_params = filter(lambda p: id(p) not in quantum_ids, self.parameters())

        optimizer = torch.optim.Adam([
            {'params': classical_params, 'lr': self.configs.lr},
            {'params': quantum_params, 'lr': self.configs.lr * 0.1}
        ])
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

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

            # Validation
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
        self.eval()
        self.to(self.device)
        preds, trues = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self(batch_x)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # Calculate standardized metrics (as per QuLTSF paper)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print_metrics(mae, mse, rmse, mape, mspe, tag="TEST RESULT")

        # Visualization of a sample (Inverted Scale)
        plt.figure(figsize=(10, 5))
        # Inverse transform only the first sample for visualization
        # Note: Scaler expects [L, M], we slice first batch, first sample
        input_real = scaler.inverse_transform(test_loader.dataset.x[0])
        true_real = scaler.inverse_transform(trues[0])
        pred_real = scaler.inverse_transform(preds[0])
        
        plt.plot(np.arange(self.seq_len), input_real[:, plot_idx], label='History', color='black')
        plt.plot(np.arange(self.seq_len, self.seq_len+self.pred_len), true_real[:, plot_idx], label='GroundTruth', color='green')
        plt.plot(np.arange(self.seq_len, self.seq_len+self.pred_len), pred_real[:, plot_idx], label='Prediction', color='red', linestyle='--')
        plt.legend()
        plt.title(f"QuLTSF Forecast (Variate {plot_idx})")
        plt.show()

        return mse, mae

    def save_model(self, scaler, folder="checkpoints", name="qultsf_model"):
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save({'state_dict': self.state_dict(), 'configs': self.configs}, f"{folder}/{name}.pth")
        joblib.dump(scaler, f"{folder}/{name}_scaler.pkl")
        print(f"Model and Scaler saved to {folder}")

    @classmethod
    def load_model(cls, folder="checkpoints", name="qltsf-original", device='cpu'):
        checkpoint = torch.load(f"{folder}/{name}.pth", map_location=device)
        configs_dict = checkpoint['configs']
        class Config: pass
        configs = Config()
        for k, v in configs_dict.items():
            setattr(configs, k, v)
        
        model = cls(configs)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = joblib.load(f"{folder}/{name}_scaler.pkl")
        return model, scaler

# ==========================================
# Sub-Module: The Hybrid Quantum-Classical Engine
# ==========================================
class Hybrid_QML_Engine(nn.Module):
    def __init__(self, lookback_window_size, forecast_window_size, num_qubits, QML_device, num_layers):
        super(Hybrid_QML_Engine, self).__init__()
        self.lookback_window_size = lookback_window_size
        self.dev = qml.device(QML_device, wires=num_qubits)

        @qml.qnode(self.dev, interface='torch', diff_method="best")
        def quantum_function(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(num_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self._quantum_circuit = quantum_function
        q_weights_shape = {'weights': (num_layers, num_qubits, 3)}
        
        self.input_classical_layer = nn.Linear(lookback_window_size, 2 ** num_qubits)
        self.hidden_quantum_layer = qml.qnn.TorchLayer(self._quantum_circuit, q_weights_shape)
        self.output_classical_layer = nn.Linear(num_qubits, forecast_window_size)

    def forward(self, x):
        B, L, M = x.shape
        # Channel Independence
        x = x.permute(0, 2, 1).reshape(B * M, L)
        
        # Processing
        x = self.input_classical_layer(x)
        # Numerical stability for AmplitudeEmbedding
        x = x + 1e-6 
        x = self.hidden_quantum_layer(x)
        x = self.output_classical_layer(x)
        
        # Reshape back to multivariate sequence
        x = x.reshape(B, M, -1).permute(0, 2, 1)
        return x