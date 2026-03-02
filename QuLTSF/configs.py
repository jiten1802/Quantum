import torch

class BaseConfig:
    # Global Settings
    data_path = 'data/weather.csv'
    seq_len = 336
    pred_len = 96
    batch_size = 16
    num_features = 21
    epochs = 30
    lr = 0.001
    QML_device = 'default.qubit' # Change to 'lightning.qubit' for speed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuLTSF_Base_Config(BaseConfig):
    model_name = 'QuLTSF_Base'
    num_qubits = 10
    num_layers = 3

class QuLTSF_Skip_Config(BaseConfig):
    model_name = 'QuLTSF_Skip'
    num_qubits = 10
    num_layers = 3

class QDLinear_Config(BaseConfig):
    model_name = 'QDLinear'
    num_qubits = 10
    num_layers = 3
    kernel_size = 25

class Patch_QuLTSF_Config(BaseConfig):
    model_name = 'Patch_QuLTSF'
    num_qubits = 4
    num_layers = 3
    patch_len = 16
    stride = 16

class Patch_QuLTSF_Skip_Config(BaseConfig):
    model_name = 'Patch_QuLTSF_Skip'
    num_qubits = 4
    num_layers = 3
    patch_len = 16
    stride = 16