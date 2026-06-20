import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "DATA" / "synthetic_dataset.csv"

def load_synthetic_data(data_path=None):
    """
    Load synthetic dataset from CSV file.
    
    Args:
        data_path: Path to synthetic_dataset.csv. If None, uses default path.
    
    Returns:
        data: numpy structured array with fields x1, x2, x3, x4, x5, y
    """
    if data_path is None:
        data_path = DATA_PATH
        if not data_path.exists():
            data_path = BASE_DIR.parent / "DATA" / "synthetic_dataset.csv"
    
    data = np.genfromtxt(data_path, delimiter=',', names=True)
    return data

def split_train_test(data, train_ratio=0.8):
    """
    Split data into train and test sets.
    First train_ratio of samples go to train, rest to test.
    """
    n_samples = len(data)
    train_size = int(n_samples * train_ratio)
    
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    return train_data, test_data

def normalize_data(train_data, test_data):
    """
    Standardize data using mean and std from training set.
    
    Args:
        train_data: numpy structured array for training
        test_data: numpy structured array for testing
    
    Returns:
        train_normalized, test_normalized: normalized arrays
        stats: dict with mean and std for each field
    """
    fields = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
    
    # Collect all feature values
    train_features = np.column_stack([train_data[f] for f in fields])
    test_features = np.column_stack([test_data[f] for f in fields])
    
    # Compute mean and std from training
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    # Normalize
    train_normalized = (train_features - mean) / std
    test_normalized = (test_features - mean) / std
    
    # Convert back to structured array
    train_result = np.zeros(len(train_normalized), dtype=[(f, float) for f in fields])
    test_result = np.zeros(len(test_normalized), dtype=[(f, float) for f in fields])
    
    for i, f in enumerate(fields):
        train_result[f] = train_normalized[:, i]
        test_result[f] = test_normalized[:, i]
    
    stats = {'mean': mean, 'std': std, 'fields': fields}
    
    return train_result, test_result, stats

def create_sequences(data, input_features, target_feature='y', seq_length=10):
    """
    Create sequences for RNN training.
    
    Args:
        data: numpy structured array
        input_features: list of feature names to use as input (e.g., ['x1', 'x2', ...])
        target_feature: name of target feature (default 'y')
        seq_length: length of input window
    
    Returns:
        X: array of shape (n_samples, seq_length, n_features)
        y: array of shape (n_samples,)
    """
    features = np.column_stack([data[f] for f in input_features])
    target = data[target_feature]
    
    n_samples = len(data) - seq_length
    X = np.zeros((n_samples, seq_length, features.shape[1]))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        X[i] = features[i:i+seq_length]
        y[i] = target[i+seq_length]  # Predict next step
    
    return X.astype(np.float32), y.astype(np.float32)

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(seq_length=10, batch_size=64, data_path=None):
    """
    Main function to get train and test dataloaders.
    
    Args:
        seq_length: input window length
        batch_size: batch size for dataloaders
        data_path: path to CSV file
    
    Returns:
        train_loader, test_loader, stats
    """
    # Load data
    data = load_synthetic_data(data_path)
    
    # Split
    train_data, test_data = split_train_test(data, train_ratio=0.8)
    
    # Normalize
    train_norm, test_norm, stats = normalize_data(train_data, test_data)
    
    # Create sequences
    input_features = ['x1', 'x2', 'x3', 'x4', 'x5']
    X_train, y_train = create_sequences(train_norm, input_features, 'y', seq_length)
    X_test, y_test = create_sequences(test_norm, input_features, 'y', seq_length)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, stats

def get_dataloaders_no_normalize(seq_length=10, batch_size=64, data_path=None, train_ratio=0.8):
    """
    Get train and test dataloaders WITHOUT normalization.
    """
    # Load data
    data = load_synthetic_data(data_path)
    
    # Split
    train_data, test_data = split_train_test(data, train_ratio)
    
    # Create sequences WITHOUT normalization
    input_features = ['x1', 'x2', 'x3', 'x4', 'x5']
    X_train, y_train = create_sequences(train_data, input_features, 'y', seq_length)
    X_test, y_test = create_sequences(test_data, input_features, 'y', seq_length)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    stats = None
    return train_loader, test_loader, stats


def get_train_test_sequences_no_normalize(seq_length=10, data_path=None, train_ratio=0.8):
    """
    Get train and test sequences WITHOUT normalization.
    Returns X_train, y_train, X_test, y_test
    """
    data = load_synthetic_data(data_path)
    train_data, test_data = split_train_test(data, train_ratio)
    
    input_features = ['x1', 'x2', 'x3', 'x4', 'x5']
    X_train, y_train = create_sequences(train_data, input_features, 'y', seq_length)
    X_test, y_test = create_sequences(test_data, input_features, 'y', seq_length)
    
    return X_train, y_train, X_test, y_test


def prepare_sequence_data(data, seq_length=10, input_features=None, target_feature='y'):
    """
    Prepare sequences for RNN training without creating DataLoaders.
    Returns X and y as numpy arrays.
    """
    if input_features is None:
        input_features = ['x1', 'x2', 'x3', 'x4', 'x5']
    
    X, y = create_sequences(data, input_features, target_feature, seq_length)
    return X, y


def get_train_test_sequences(seq_length=10, data_path=None, train_ratio=0.8):
    """
    Get train and test sequences as numpy arrays.
    Returns X_train, y_train, X_test, y_test, stats
    """
    data = load_synthetic_data(data_path)
    train_data, test_data = split_train_test(data, train_ratio)
    train_norm, test_norm, stats = normalize_data(train_data, test_data)
    
    input_features = ['x1', 'x2', 'x3', 'x4', 'x5']
    X_train, y_train = create_sequences(train_norm, input_features, 'y', seq_length)
    X_test, y_test = create_sequences(test_norm, input_features, 'y', seq_length)
    
    return X_train, y_train, X_test, y_test, stats
