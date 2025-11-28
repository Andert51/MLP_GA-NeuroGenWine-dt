"""
VinoGen-CyberCore: Utility Functions
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any
import json
import time
from functools import wraps


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def timer(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"⏱️  {func.__name__} executed in {elapsed:.2f} seconds")
        return result
    return wrapper


def save_json(data: Dict[str, Any], filepath: Path):
    """Save dictionary to JSON file."""
    filepath.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy types to Python types recursively
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        return obj
    
    cleaned_data = convert(data)
    
    with open(filepath, 'w') as f:
        json.dump(cleaned_data, f, indent=2)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file to dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_number(num: float, precision: int = 4) -> str:
    """Format number for display."""
    if abs(num) >= 1000:
        return f"{num:,.{precision}f}"
    return f"{num:.{precision}f}"


def calculate_model_complexity(genome: Dict) -> int:
    """
    Calculate the complexity (parameter count) of a network architecture.
    
    Args:
        genome: Network genome dictionary
        
    Returns:
        Total parameter count
    """
    layers = genome['hidden_layers']
    
    # Assume some input/output dimensions
    input_dim = 11  # Wine dataset features
    output_dim = 1
    
    total_params = 0
    prev_dim = input_dim
    
    for neurons in layers:
        # Weights + biases
        total_params += prev_dim * neurons + neurons
        prev_dim = neurons
    
    # Output layer
    total_params += prev_dim * output_dim + output_dim
    
    return total_params


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def ensure_dir(directory: Path):
    """Ensure directory exists."""
    directory.mkdir(exist_ok=True, parents=True)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Validation loss
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop


def create_ascii_art(text: str, style: str = 'banner') -> str:
    """
    Create ASCII art from text.
    
    Args:
        text: Text to convert
        style: ASCII art style
        
    Returns:
        ASCII art string
    """
    try:
        import pyfiglet
        return pyfiglet.figlet_format(text, font=style)
    except:
        return text


def bytes_to_human_readable(bytes_size: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"
