import torch
from torch.utils.data import DataLoader
from .preprocessing import WeatherDataset, preprocess_weather_data

def data_provider(configs, flag='train'):
    """
    Provides DataLoaders for specific experiment phases.
    
    Args:
        configs: Configuration object containing data_path, seq_len, pred_len, batch_size.
        flag: 'train', 'val', or 'test'
        
    Returns:
        loader: PyTorch DataLoader object
        data_dict: The full dictionary containing scalers and metadata (only returned for 'train')
    """
    
    # 1. Run full preprocessing pipeline
    # We do this once to ensure consistent scaling across all splits
    data_dict = preprocess_weather_data(
        csv_path=configs.data_path,
        seq_len=configs.seq_len,
        pred_len=configs.pred_len
    )

    # 2. Select the correct split based on the flag
    if flag == 'train':
        shuffle = True
        drop_last = True
        data_x, data_y = data_dict['train']
    elif flag == 'val':
        shuffle = False
        drop_last = True
        data_x, data_y = data_dict['val']
    else: # test
        shuffle = False
        drop_last = False
        data_x, data_y = data_dict['test']

    # 3. Create Dataset instance
    dataset = WeatherDataset(data_x, data_y)

    # 4. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=shuffle,
        num_workers=configs.num_workers if hasattr(configs, 'num_workers') else 0,
        drop_last=drop_last
    )

    # For the training phase, we return the data_dict so the user can access the scaler later
    if flag == 'train':
        return loader, data_dict
    
    return loader

def get_all_loaders(configs):
    """
    Utility to get all three loaders in one call.
    Useful for main.py execution.
    """
    # Get Train Loader and the metadata dictionary
    train_loader, data_dict = data_provider(configs, flag='train')
    
    # Get Val and Test loaders
    val_loader = data_provider(configs, flag='val')
    test_loader = data_provider(configs, flag='test')
    
    return train_loader, val_loader, test_loader, data_dict