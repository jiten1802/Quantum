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

class Patch_QuLTSF_Skip_Model(nn.Module):
    def __init__(self, configs):
        super(Patch_QuLTSF_Skip_Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layers = configs.num_layers
        self.QML_device = configs.QML_device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Patching Config (P=16 is optimal for 4-qubit Amplitude Embedding)
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 16)

        # 1. Reversible Instance Normalization (RevIN)
        num_features = getattr(configs, 'num_features', 21)
        self.revin_layer = RevIN(num_features)

        # 2. Patch-based Quantum Engine with Skip Connection
        self.engine = Patch_QML_Skip_Engine(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            patch_len=self.patch_len,
            stride=self.stride,
            num_layers=self.num_layers,
            QML_device=self.QML_device
        )

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Variates]
        """
        # A. Instance Normalization
        x = self.revin_layer(x, 'norm')
        
        # B. Forward through Engine (Quantum Path + Skip Path)
        x = self.engine(x)
        
        # C. Denormalization
        x = self.revin_layer(x, 'denorm')
        return x

    def summary(self):
        """Prints a summary including patching logic, residual logic, and parameters."""
        print("\n" + "="*90)
        print(f"{'Patch-QuLTSF + Skip Connection Model Summary':^90}")
        print("="*90)
        print(f"Logic Configuration:")
        print(f"  Input Seq: {self.seq_len} | Pred Seq: {self.pred_len}")
        print(f"  Patching:  {self.engine.num_patches} patches of length {self.patch_len}")
        print(f"  Residual:  Global Skip Connection (Linear {self.seq_len}->{self.pred_len})")
        print("-" * 90)
        print(f"{'Layer / Parameter Name':<55} | {'Shape':<20} | {'Params'}")
        print("-" * 90)
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                p_count = param.numel()
                total_params += p_count
                print(f"{name:<55} | {str(list(param.shape)):<20} | {p_count:,}")
        print("-" * 90)
        print(f"Total Trainable Parameters: {total_params:,}")
        print("="*90 + "\n")

    def train_model(self, train_loader, val_loader):
        self.to(self.device)
        criterion = nn.MSELoss()
        
        # Separated Learning Rates
        quantum_params = list(self.engine.patch_encoder.parameters())
        quantum_ids = list(map(id, quantum_params))
        classical_params = filter(lambda p: id(p) not in quantum_ids, self.parameters())

        optimizer = torch.optim.Adam([
            {'params': classical_params, 'lr': self.configs.lr},
            {'params': quantum_params, 'lr': self.configs.lr * 0.1} # Slower Quantum LR
        ])
        
        # Patience set to 5 to allow Quantum layers to stabilize alongside the skip connection
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
        print_metrics(mae, mse, rmse, mape, mspe, tag="PATCH+SKIP TEST")

        # Visualization
        plt.figure(figsize=(10, 5))
        input_real = scaler.inverse_transform(test_loader.dataset.x[0])
        true_real = scaler.inverse_transform(trues[0])
        pred_real = scaler.inverse_transform(preds[0])
        plt.plot(input_real[:, plot_idx], label='History', color='black')
        plt.plot(np.arange(self.seq_len, self.seq_len+self.pred_len), true_real[:, plot_idx], label='True', color='green')
        plt.plot(np.arange(self.seq_len, self.seq_len+self.pred_len), pred_real[:, plot_idx], label='P-Skip-Pred', color='red', ls='--')
        plt.legend(); plt.title(f"Patch+Skip Forecast (Variate {plot_idx})"); plt.show()
        return mse, mae

    def save_model(self, scaler, folder="checkpoints", name="patch_qultsf_skip"):
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save({'state_dict': self.state_dict(), 'configs': self.configs}, f"{folder}/{name}.pth")
        joblib.dump(scaler, f"{folder}/{name}_scaler.pkl")

    @classmethod
    def load_model(cls, folder="checkpoints", name="patch_qultsf_skip", device='cpu'):
        checkpoint = torch.load(f"{folder}/{name}.pth", map_location=device)
        model = cls(checkpoint['configs'])
        model.load_state_dict(checkpoint['state_dict'])
        scaler = joblib.load(f"{folder}/{name}_scaler.pkl")
        return model, scaler

# ==========================================
# Core Engine: Patching + Skip Path
# ==========================================
class Patch_QML_Skip_Engine(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, num_layers, QML_device):
        super(Patch_QML_Skip_Engine, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.num_qubits = 4 
        
        # --- QUANTUM BRANCH ---
        self.dev = qml.device(QML_device, wires=self.num_qubits)

        @qml.qnode(self.dev, interface='torch', diff_method="best")
        def quantum_function(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.patch_encoder = qml.qnn.TorchLayer(quantum_function, {'weights': (num_layers, self.num_qubits, 3)})
        
        # Quantum Projection Head
        self.head = nn.Linear(self.num_patches * self.num_qubits, pred_len)

        # --- CLASSICAL SKIP BRANCH ---
        self.skip_connection = nn.Linear(seq_len, pred_len)
        # Init skip connection to be small to force Quantum path to engage
        nn.init.normal_(self.skip_connection.weight, mean=0, std=0.01)

    def forward(self, x):
        B, L, M = x.shape
        # Channel Independence
        x_flat = x.permute(0, 2, 1).reshape(B * M, L)
        
        # 1. Quantum Path
        # Patching
        x_patches = x_flat.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x_p_stacked = x_patches.reshape(-1, self.patch_len)
        x_p_stacked = x_p_stacked + 1e-6 # Numerical stability
        
        # Encode patches
        enc = self.patch_encoder(x_p_stacked)
        enc = enc.reshape(B * M, self.num_patches, self.num_qubits)
        q_out = self.head(enc.reshape(B * M, -1))
        
        # 2. Skip Path
        c_out = self.skip_connection(x_flat)
        
        # 3. Sum & Restore Shape
        out = q_out + c_out
        out = out.reshape(B, M, -1).permute(0, 2, 1)
        return out