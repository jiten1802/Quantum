def run_inference(model_class, test_loader, experiment_name, device, plot_idx=0):
    """
    Standard function to load and test any of the 5 models.
    Pass the CLASS (e.g. Patch_QuLTSF_Model) as the first argument.
    """
    print(f"\n" + "="*50)
    print(f"STARTING INFERENCE: {experiment_name}")
    print("="*50)
    
    # 1. Load the model using the @classmethod we defined in each file
    # This automatically reconstructs the architecture from saved configs
    loaded_model, loaded_scaler = model_class.load_model()
    
    # 2. Trigger the internal testing logic
    # (Calculates MSE/MAE and shows the Matplotlib plot)
    mse, mae = loaded_model.test_model(
        test_loader=test_loader, 
        scaler=loaded_scaler, 
        plot_idx=plot_idx
    )
    
    print(f"\n📈 Final Metrics for {experiment_name}:")
    print(f"Test MSE: {mse:.5f} | Test MAE: {mae:.5f}")
    
    return loaded_model
