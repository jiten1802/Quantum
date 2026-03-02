import numpy as np
import torch

def RSE(pred, true):
    """Relative Squared Error"""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    """Correlation Coefficient"""
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    """Mean Absolute Error - Standard metric"""
    return np.mean(np.abs(pred - true))
    
def MSE(pred, true):
    """Mean Squared Error - Primary metric"""
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    """Root Mean Squared Error"""
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    """Mean Squared Percentage Error"""
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    """
    Calculates all major metrics at once.
    
    Args:
        pred: Predicted values (NumPy array)
        true: Ground truth values (NumPy array)
        
    Returns:
        tuple: (mae, mse, rmse, mape, mspe)
    """
    # Ensure inputs are numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
        
    # Handle potential zeros in MAPE/MSPE by adding a tiny epsilon
    true = np.where(true == 0, 1e-5, true)

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def print_metrics(mae, mse, rmse, mape, mspe, tag="TEST"):
    """Formatted print for metrics results."""
    print(f"\n>>>> {tag} METRICS <<<<")
    print(f"MSE:  {mse:.5f}")
    print(f"MAE:  {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"MAPE: {mape:.5f}")
    print(f"MSPE: {mspe:.5f}")
    print("="*20)