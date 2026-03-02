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

class Patch_QuLTSF_Model(nn.Module):
    def __init__(self, configs):
        super(Patch_QuLTSF_Model, self).__init__()
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

        # 2. Patch-based Quantum Engine
        self.patch_qml_engine = Patch_QML_Engine(
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
        
        # B. Patching & Quantum Processing
        x = self.patch_qml_engine(x)
        
        # C. Denormalization
        x = self.revin_layer(x, 'denorm')
        return x

    def summary(self):
        """Prints a summary including patching logic and parameters."""
        print("\n" + "="*85)
        print(f"{'Patch-QuLTSF Model Summary':^85}")
        print("="*85)
        print(f"Patching Logic:")
        print(f"  Sequence Length: {self.seq_len} | Patch Length: {self.patch_len} | Stride: {self.stride}")
        print(f"  Total Patches:   {self.patch_qml_engine.num_patches} | Qubits/Patch: {self.patch_qml_engine.num_qubits}")
        print("-" * 85)
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
        
        # Identify Quantum vs Classical parameters for separate LRs
        quantum_params = list(self.patch_qml_engine.patch_encoder.parameters())
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
        print_metrics(mae, mse, rmse, mape, mspe, tag="PATCH-TEST")

        # Visuals
        plt.figure(figsize=(10, 5))
        input_real = scaler.inverse_transform(test_loader.dataset.x[0])
        true_real = scaler.inverse_transform(trues[0])
        pred_real = scaler.inverse_transform(preds[0])
        plt.plot(input_real[:, plot_idx], label='History', color='black')
        plt.plot(np.arange(self.seq_len, self.seq_len+self.pred_len), true_real[:, plot_idx], label='True', color='green')
        plt.plot(np.arange(self.seq_len, self.seq_len+self.pred_len), pred_real[:, plot_idx], label='Patch-Prediction', color='red', ls='--')
        plt.legend(); plt.title(f"Patch-QuLTSF Forecast (Variate {plot_idx})"); plt.show()
        return mse, mae

    def save_model(self, scaler, folder="checkpoints", name="patch_qultsf"):
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save({'state_dict': self.state_dict(), 'configs': self.configs}, f"{folder}/{name}.pth")
        joblib.dump(scaler, f"{folder}/{name}_scaler.pkl")

    @classmethod
    def load_model(cls, folder="checkpoints", name="patch_qultsf", device='cpu'):
        checkpoint = torch.load(f"{folder}/{name}.pth", map_location=device)
        model = cls(checkpoint['configs'])
        model.load_state_dict(checkpoint['state_dict'])
        scaler = joblib.load(f"{folder}/{name}_scaler.pkl")
        return model, scaler

# ==========================================
# Sub-Module: The Patching Quantum Engine
# ==========================================
class Patch_QML_Engine(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, num_layers, QML_device):
        super(Patch_QML_Engine, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.num_qubits = 4 # Fixed for 16-length patch (2^4 = 16)
        
        self.dev = qml.device(QML_device, wires=self.num_qubits)

        @qml.qnode(self.dev, interface='torch', diff_method="best")
        def quantum_function(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.num_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        q_weights_shape = {'weights': (num_layers, self.num_qubits, 3)}
        self.patch_encoder = qml.qnn.TorchLayer(quantum_function, q_weights_shape)
        
        # Projection Head: (Num_Patches * 4 qubits) -> pred_len
        self.head = nn.Linear(self.num_patches * self.num_qubits, pred_len)

    def forward(self, x):
        B, L, M = x.shape
        # Channel Independence
        x = x.permute(0, 2, 1).reshape(B * M, L)
        
        # Patching (Unfold)
        # [BM, L] -> [BM, Num_Patches, Patch_Len]
        x_patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        # Parallel processing of all patches across all samples
        # [BM * Num_Patches, Patch_Len]
        x_p_stacked = x_patches.reshape(-1, self.patch_len)
        x_p_stacked = x_p_stacked + 1e-6 # Stability
        
        # Quantum Encode: [N, 16] -> [N, 4]
        enc = self.patch_encoder(x_p_stacked)
        
        # Flatten and Project
        enc = enc.reshape(B * M, self.num_patches, self.num_qubits)
        out = self.head(enc.reshape(B * M, -1))
        
        # Restore Shape
        out = out.reshape(B, M, -1).permute(0, 2, 1)
        return out