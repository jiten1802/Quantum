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

class QuLTSF_Skip_Model(nn.Module):
    def __init__(self, configs):
        super(QuLTSF_Skip_Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_qubits = configs.num_qubits
        self.num_layers = configs.num_layers
        self.QML_device = configs.QML_device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Reversible Instance Normalization (RevIN)
        # Prevents absolute value drift; crucial for Quantum Amplitude Embedding stability
        num_features = getattr(configs, 'num_features', 21)
        self.revin_layer = RevIN(num_features)

        # 2. Hybrid QML Model with Skip Connection
        self.hybrid_qml_model = Hybrid_QML_Model(
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
        
        # B. Forward through Hybrid Engine (Quantum + Skip Path)
        x = self.hybrid_qml_model(x)
        
        # C. Denormalization
        x = self.revin_layer(x, 'denorm')
        return x

    def summary(self):
        """Prints a summary of the model architecture and parameters."""
        print("\n" + "="*85)
        print(f"{'QuLTSF + Skip Connection Model Summary':^85}")
        print("="*85)
        print(f"{'Layer / Parameter Name':<50} | {'Shape':<20} | {'Params'}")
        print("-" * 85)
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                p_count = param.numel()
                total_params += p_count
                print(f"{name:<50} | {str(list(param.shape)):<20} | {p_count:,}")
        print("-" * 85)
        print(f"Total Trainable Parameters: {total_params:,}")
        print("="*85 + "\n")

    def train_model(self, train_loader, val_loader):
        self.to(self.device)
        criterion = nn.MSELoss()
        
        # Split parameters to use different Learning Rates
        # Quantum parameters are sensitive; Classical layers (Skip/Linears) can learn faster
        quantum_params = list(self.hybrid_qml_model.hidden_quantum_layer.parameters())
        quantum_ids = list(map(id, quantum_params))
        classical_params = filter(lambda p: id(p) not in quantum_ids, self.parameters())

        optimizer = torch.optim.Adam([
            {'params': classical_params, 'lr': self.configs.lr},
            {'params': quantum_params, 'lr': self.configs.lr * 0.1} 
        ])
        
        # Patience set to 5 to allow Quantum layers time to overcome Barren Plateaus
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
                # Clip gradients to prevent explosion in Quantum path
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
        
        # Metric calculation (MSE, MAE, etc.)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print_metrics(mae, mse, rmse, mape, mspe, tag="TEST RESULT")

        # Visualization
        plt.figure(figsize=(10, 5))
        # Get one sample for plotting
        input_real = scaler.inverse_transform(test_loader.dataset.x[0])
        true_real = scaler.inverse_transform(trues[0])
        pred_real = scaler.inverse_transform(preds[0])
        
        plt.plot(np.arange(self.seq_len), input_real[:, plot_idx], label='History', color='black')
        plt.plot(np.arange(self.seq_len, self.seq_len+self.pred_len), true_real[:, plot_idx], label='GroundTruth', color='green')
        plt.plot(np.arange(self.seq_len, self.seq_len+self.pred_len), pred_real[:, plot_idx], label='Prediction', color='red', linestyle='--')
        plt.legend()
        plt.title(f"QuLTSF+Skip Forecast (Variate Index {plot_idx})")
        plt.show()

        return mse, mae

    def save_model(self, scaler, folder="checkpoints", name="qultsf_skip"):
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save({'state_dict': self.state_dict(), 'configs': self.configs}, f"{folder}/{name}.pth")
        joblib.dump(scaler, f"{folder}/{name}_scaler.pkl")
        print(f"Checkpoint saved to {folder}/{name}.pth")

    @classmethod
    def load_model(cls, folder="checkpoints", name="qultsf_skip", device='cpu'):
        checkpoint = torch.load(f"{folder}/{name}.pth", map_location=device)
        model = cls(checkpoint['configs'])
        model.load_state_dict(checkpoint['state_dict'])
        scaler = joblib.load(f"{folder}/{name}_scaler.pkl")
        return model, scaler

# ==========================================
# Core Engine: Hybrid_QML_Model
# ==========================================
class Hybrid_QML_Model(nn.Module):
    def __init__(self, lookback_window_size, forecast_window_size, num_qubits, QML_device, num_layers):
        super(Hybrid_QML_Model, self).__init__()
        self.lookback_window_size = lookback_window_size
        self.num_qubits = num_qubits
        
        # --- QUANTUM BRANCH ---
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

        # --- CLASSICAL SKIP CONNECTION ---
        self.skip_connection = nn.Linear(lookback_window_size, forecast_window_size)
        
        # Initialize skip connection to be small at start to encourage Quantum path learning
        nn.init.normal_(self.skip_connection.weight, mean=0, std=0.01)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Variates]
        B, L, M = x.shape
        
        # Channel Independence: Flatten to [Batch * Variates, Seq_Len]
        x_flat = x.permute(0, 2, 1).reshape(B * M, L)

        # 1. Quantum Path
        q_out = self.input_classical_layer(x_flat)
        q_out = q_out + 1e-6 # Numerical stability for Amplitude Embedding
        q_out = self.hidden_quantum_layer(q_out)
        q_out = self.output_classical_layer(q_out)
        
        # 2. Skip Connection Path
        c_out = self.skip_connection(x_flat)
        
        # 3. Combine Paths
        final_output = q_out + c_out
        
        # 4. Reshape back to [Batch, Pred_Len, Variates]
        final_output = final_output.reshape(B, M, -1).permute(0, 2, 1)
        
        return final_output