# QuLTSF: Quantum Long-Term Time-Series Forecasting

This repository explores hybrid quantum-classical models for long-term,
multivariate time-series forecasting. It contains six forecasting variants,
shared preprocessing and evaluation utilities, a weather dataset, experiment
notebooks, reference literature, and architecture figures.

The default experiment uses 336 historical time steps to forecast the next 96
steps for all 21 variables in the bundled weather dataset.

## Models

| Model | Class | Description |
| --- | --- | --- |
| QuLTSF | `QuLTSF_Model` | Projects each variable's history into an amplitude-encoded quantum circuit. |
| QuLTSF + Skip | `QuLTSF_Skip_Model` | Adds a classical linear residual path to the full-window quantum model. |
| QDLinear | `QuLTSF_Decomp_Model` | Uses a classical trend branch and quantum seasonal branch after decomposition. |
| Patch-QuLTSF | `Patch_QuLTSF_Model` | Divides the input into 16-step patches encoded with four qubits. |
| Patch-QuLTSF + Skip | `Patch_QuLTSF_Skip_Model` | Adds a global classical skip connection to Patch-QuLTSF. |
| Stiefel-QuLTSF | `Stiefel_QuLTSF_Model` | Learns complex unitary matrices using Cayley updates on the Stiefel manifold. |

All models use channel-independent processing and Reversible Instance
Normalization (RevIN). Their outputs have shape:

```text
[batch size, forecast length, number of variables]
```

## Repository layout

```text
Quantum/
├── QuLTSF/
│   ├── configs.py
│   ├── data/weather.csv
│   ├── models/
│   │   ├── base_model.py
│   │   ├── qultsf_skip.py
│   │   ├── qdlinear.py
│   │   ├── patched_qultsf.py
│   │   ├── patch_qultsf_skip.py
│   │   └── stiefel_qultsf.py
│   ├── utils/
│   │   ├── preprocessing.py
│   │   ├── dataloaders.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── metrics.py
│   ├── training_and_inference.ipynb
│   └── training-the-unitary-matrix-for-qultsf.ipynb
├── Literature/
├── pics/
├── requirements.txt
└── README.md
```

Generated checkpoints are stored in `QuLTSF/checkpoints/`. The folder is
created automatically when training finishes.

## Setup

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install jupyterlab
```

### Linux or macOS

```bash
python3 -m venv myenv
source myenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install jupyterlab
```

The main dependencies are PyTorch, PennyLane, pandas, NumPy, scikit-learn,
Matplotlib, and joblib.

## Training and inference

Paths in the configuration are relative to `QuLTSF`, so start Jupyter from
that directory.

### Windows PowerShell

```powershell
cd QuLTSF
..\myenv\Scripts\python.exe -m jupyter lab training_and_inference.ipynb
```

### Linux or macOS

```bash
cd QuLTSF
../myenv/bin/python -m jupyter lab training_and_inference.ipynb
```

The notebook has an independent section for each model. To run one model:

1. Run the import cell at the top.
2. Run the model's data-loading cell.
3. Run its model construction and training cell.
4. Run its inference cell after training finishes.

Training is handled by `run_training_pipeline()`. It calls the model's
`train_model()` method and automatically saves the trained weights and fitted
data scaler.

For example, the Stiefel model uses:

```python
stiefel_configs = Stiefel_QuLTSF_Config()
train_loader, val_loader, test_loader, data_dict = get_all_loaders(
    stiefel_configs
)

stiefel_model = Stiefel_QuLTSF_Model(stiefel_configs)

run_training_pipeline(
    model=stiefel_model,
    train_loader=train_loader,
    val_loader=val_loader,
    scaler=data_dict["scaler"],
    experiment_name="weather_stiefel_qultsf",
)
```

This creates:

```text
QuLTSF/checkpoints/weather_stiefel_qultsf.pth
QuLTSF/checkpoints/weather_stiefel_qultsf_scaler.pkl
```

Load and evaluate the checkpoint with:

```python
run_inference(
    model_class=Stiefel_QuLTSF_Model,
    test_loader=test_loader,
    experiment_name="weather_stiefel_qultsf",
    device=stiefel_configs.device,
    plot_idx=1,
)
```

The `experiment_name` used for inference must exactly match the name used for
training. Training again with the same name overwrites that checkpoint.

## Configuration

Experiment settings are defined in `QuLTSF/configs.py`. Common defaults are:

```python
seq_len = 336
pred_len = 96
batch_size = 16
num_features = 21
epochs = 30
lr = 0.001
QML_device = "default.qubit"
```

Change settings before creating the data loaders and model:

```python
configs = Patch_QuLTSF_Config()
configs.epochs = 10
configs.batch_size = 8
```

For PennyLane installations that provide it, `lightning.qubit` may be used as
a faster simulator:

```python
configs.QML_device = "lightning.qubit"
```

## Data pipeline

The shared preprocessing pipeline:

1. Loads `QuLTSF/data/weather.csv`.
2. Parses the date column and forward-fills missing values.
3. Splits observations chronologically into 70% training, 10% validation, and
   20% testing data.
4. Fits `StandardScaler` on training data only.
5. Creates sliding input and forecast windows.
6. Wraps the arrays in PyTorch data loaders.

RevIN is applied inside each model before forecasting and reversed at the
model output.

## Evaluation

Inference reports MSE, MAE, RMSE, MAPE, and MSPE on standardized values. It
also plots one selected variable in its original scale. Change `plot_idx` to
visualize a different weather variable.

## Stiefel model notes

The Stiefel model projects each input history into a complex state vector,
applies trainable unitary matrices, measures element-wise intensity, and maps
the result to the forecast horizon. Classical layers use Adam. Complex
matrices use a Neumann-approximated Cayley update designed to keep them close
to the unitary manifold.

The default 10-qubit configuration has state dimension 1,024. Four complex
`1024 x 1024` matrices, their gradients, and Cayley-update intermediates need
substantial memory and computation. GPU execution is strongly recommended.

For a quick smoke test:

```python
stiefel_configs = Stiefel_QuLTSF_Config()
stiefel_configs.num_qubits = 4
stiefel_configs.num_layers = 1
stiefel_configs.epochs = 1
```

Checkpoints retain the configuration used for training and reconstruct the
correct architecture automatically during loading.

## Adding another model

Models compatible with the shared pipeline provide these methods:

```python
train_model(train_loader, val_loader)
test_model(test_loader, scaler, plot_idx=0)
save_model(scaler, folder="checkpoints", name="model_name")
load_model(folder="checkpoints", name="model_name", device="cpu")
```

Add a matching configuration class in `configs.py`, then add training and
inference cells to `training_and_inference.ipynb`.

## License

This project is distributed under the terms in [LICENSE](LICENSE).
