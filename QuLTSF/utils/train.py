def run_training_pipeline(model, train_loader, val_loader, scaler, experiment_name):
    """
    Standard function to train and save any of the 5 models.
    """
    print(f"\n" + "="*50)
    print(f"STARTING TRAINING: {experiment_name}")
    print("="*50)
    
    # 1. Trigger the internal training logic
    # (Handles optimizer splitting, scheduler, and epoch loops)
    model.train_model(train_loader, val_loader)
    
    # 2. Save the final weights and the scaler
    # This creates {experiment_name}.pth and {experiment_name}_scaler.pkl
    model.save_model(scaler, name=experiment_name)
    
    print(f"\n✅ Training Complete. Model saved as: {experiment_name}")
