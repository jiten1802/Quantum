import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Reversible Instance Normalization (RevIN)
# ==========================================
class RevIN(nn.Module):
    """
    Stabilizes training by normalizing each input window individually.
    Essential for Quantum models to prevent absolute value saturation.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise NotImplementedError


# ==========================================
# 2. PyTorch Dataset Wrapper
# ==========================================
class WeatherDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.FloatTensor(x_data)
        self.y = torch.FloatTensor(y_data)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ==========================================
# 3. Main Preprocessing Pipeline
# ==========================================
def create_sequences(dataset, seq_len, pred_len):
    """Helper function to create sliding windows."""
    X, Y = [], []
    for i in range(len(dataset) - seq_len - pred_len + 1):
        X.append(dataset[i : i + seq_len])
        Y.append(dataset[i + seq_len : i + seq_len + pred_len])
    return np.array(X), np.array(Y)

def preprocess_weather_data(csv_path, seq_len=336, pred_len=96):
    """
    Loads, cleans, scales, and windows the weather data.
    Ratios: 70% Train, 10% Val, 20% Test.
    """
    # 1. Load Data
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # 2. Handle Missing Values
    if df.isnull().values.any():
        # Modern pandas ffill replaces method='ffill'
        df = df.ffill().fillna(0)
    
    data = df.values.astype(np.float32)
    print(f"Dataset Loaded: {data.shape}")

    # 3. Chronological Split
    n = len(data)
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test

    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val:]

    # 4. Global Scaling (StandardScaler)
    scaler = StandardScaler()
    scaler.fit(train_data)

    train_sc = scaler.transform(train_data)
    val_sc = scaler.transform(val_data)
    test_sc = scaler.transform(test_data)

    # 5. Windowing
    x_train, y_train = create_sequences(train_sc, seq_len, pred_len)
    x_val, y_val = create_sequences(val_sc, seq_len, pred_len)
    x_test, y_test = create_sequences(test_sc, seq_len, pred_len)
    
    # Verification
    if np.isnan(x_train).any():
        raise ValueError("Critical Error: NaNs detected in sequences!")

    print(f"Preprocessing Complete: {x_train.shape[0]} training samples generated.")
    
    return {
        'train': (x_train, y_train),
        'val': (x_val, y_val),
        'test': (x_test, y_test),
        'scaler': scaler,
        'num_features': data.shape[1]
    }

def get_dataloaders(data_dict, batch_size=16):
    """Helper to generate Torch DataLoaders directly from the data_dict."""
    train_loader = DataLoader(
        WeatherDataset(*data_dict['train']), 
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        WeatherDataset(*data_dict['val']), 
        batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        WeatherDataset(*data_dict['test']), 
        batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, val_loader, test_loader