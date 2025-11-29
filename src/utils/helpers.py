"""
VinoGen-CyberCore: Enhanced Utility Functions
Robust data handling, model persistence, and file operations.
"""

import numpy as np
import torch
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import time
from functools import wraps
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


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
    """Save dictionary to JSON file with robust type conversion."""
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
        elif hasattr(obj, '__class__') and 'torch' in obj.__class__.__module__:
            # Handle torch tensors
            return obj.detach().cpu().numpy().tolist() if hasattr(obj, 'detach') else obj
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


def save_model(model, genome: Dict, history: Dict, metrics: Dict, 
               output_dir: Path, filename: Optional[str] = None) -> str:
    """
    Save trained model with complete metadata.
    
    Args:
        model: Trained PyTorch model
        genome: Network architecture genome
        history: Training history
        metrics: Performance metrics
        output_dir: Output directory
        filename: Optional custom filename
        
    Returns:
        Path to saved model
    """
    from pathlib import Path
    output_dir = Path(output_dir)  # Convert to Path if string
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_best_genome.pkl"
    
    filepath = models_dir / filename
    
    # Package everything
    # Get device from model parameters
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    
    # Extract fitness from genome (Genome object or dict)
    if hasattr(genome, 'fitness'):
        fitness = genome.fitness
    else:
        fitness = genome.get('fitness', 0.0)
    
    # Extract accuracy from metrics
    test_accuracy = metrics.get('accuracy', metrics.get('test_accuracy', 0.0))
    
    package = {
        'model_state_dict': model.state_dict(),
        'genome': genome,
        'history': history,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'input_dim': model.input_dim,
        'output_dim': model.output_dim,
        'device': str(device),
        'fitness': fitness,
        'test_accuracy': test_accuracy
    }
    
    # Save with joblib for better compression
    joblib.dump(package, filepath, compress=3)
    
    return str(filepath)


def load_model(filepath: Path, model_class):
    """
    Load saved model package.
    
    Args:
        filepath: Path to saved model
        model_class: Model class to instantiate
        
    Returns:
        Tuple of (model, genome, history, metrics)
    """
    from pathlib import Path
    filepath = Path(filepath)  # Convert to Path if string
    package = joblib.load(filepath)
    
    # Reconstruct model
    model = model_class(
        input_dim=package['input_dim'],
        output_dim=package['output_dim'],
        genome=package['genome']
    )
    
    model.load_state_dict(package['model_state_dict'])
    model.eval()
    
    return model, package['genome'], package['history'], package['metrics']


def list_saved_models(output_dir: Path) -> list:
    """List all saved models in the output directory."""
    from pathlib import Path
    output_dir = Path(output_dir)  # Convert to Path if string
    models_dir = output_dir / "models"
    if not models_dir.exists():
        return []
    
    models = []
    for filepath in models_dir.glob("*.pkl"):
        try:
            package = joblib.load(filepath)
            models.append({
                'filepath': str(filepath),
                'filename': filepath.name,
                'timestamp': package.get('timestamp', 'Unknown'),
                'fitness': package.get('fitness', 0.0),
                'accuracy': package.get('test_accuracy', 0.0)
            })
        except Exception:
            continue
    
    return sorted(models, key=lambda x: x['timestamp'], reverse=True)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (ZeroDivisionError, TypeError):
        return default


def create_markdown_report(results: Dict, genome, output_path: Path) -> str:
    """
    Generate a comprehensive Markdown mission report.
    
    Args:
        results: Results dictionary
        genome: Best genome (Genome object or dict)
        output_path: Output file path
        
    Returns:
        Path to generated report
    """
    from pathlib import Path
    output_path = Path(output_path)  # Convert to Path if string
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Handle both Genome objects and dicts
    if hasattr(genome, 'hidden_layers'):
        # It's a Genome object
        hidden_layers = genome.hidden_layers
        activation_functions = genome.activation_functions
        learning_rate = genome.learning_rate
    else:
        # It's a dict
        hidden_layers = genome['hidden_layers']
        activation_functions = genome['activation_functions']
        learning_rate = genome.get('learning_rate', 0.001)
    
    # Build architecture string
    arch_str = " → ".join(str(n) for n in hidden_layers)
    act_str = ", ".join(activation_functions)
    
    report = f"""# 🧬 VINOGEN-CYBERCORE MISSION REPORT
**Neural Architecture Evolution Complete**

---

## 📊 EXECUTIVE SUMMARY

**Mission Timestamp:** {timestamp}
**Status:** ✅ COMPLETE
**Task:** {results.get('task', 'classification').upper()}

---

## 🏗️ EVOLVED NEURAL ARCHITECTURE

**Topology:** `{arch_str}`
**Activations:** `{act_str}`
**Learning Rate:** `{learning_rate:.6f}`
**Total Parameters:** `{results.get('total_parameters', 'N/A')}`

---

## 📈 PERFORMANCE METRICS

### Training Results
- **Best Fitness:** `{results.get('best_fitness', 0.0):.6f}`
- **Generations:** `{results.get('generations', 0)}`
- **Population Size:** `{results.get('population_size', 0)}`

### Test Performance
- **Test Loss:** `{results.get('test_loss', 0.0):.6f}`
- **Test Accuracy:** `{results.get('test_accuracy', 0.0):.6f}` ({results.get('test_accuracy', 0.0) * 100:.2f}%)

### Training Time
- **Total Duration:** `{results.get('training_time', 0):.2f}` seconds
- **Time per Generation:** `{safe_divide(results.get('training_time', 0), results.get('generations', 1)):.2f}` seconds

---

## 🎯 CLASSIFICATION ANALYSIS

```
{results.get('classification_report', 'N/A')}
```

---

## 📁 GENERATED ASSETS

### Visualizations
- ✅ Network Topology: `output/network_topology.png`
- ✅ Activation Flow: `output/network_activation.gif`
- ✅ Learning Curves: `output/learning_curves.png`
- ✅ Confusion Matrix: `output/confusion_matrix.png`
- ✅ Evolution History: `output/evolution_history.png`
- ✅ 3D Loss Landscape: `output/loss_landscape_3d.html`

### Models
- ✅ Best Model: `{results.get('model_path', 'output/models/best_genome.pkl')}`

### Reports
- ✅ JSON Results: `output/results.json`
- ✅ Text Report: `output/final_report.txt`

---

## 🔬 TECHNICAL DETAILS

### Genetic Algorithm Configuration
- **Mutation Rate:** `{results.get('mutation_rate', 0.3)}`
- **Crossover Rate:** `{results.get('crossover_rate', 0.7)}`
- **Elitism Ratio:** `{results.get('elitism_ratio', 0.2)}`

### Dataset Information
- **Training Samples:** `{results.get('train_samples', 0)}`
- **Validation Samples:** `{results.get('val_samples', 0)}`
- **Test Samples:** `{results.get('test_samples', 0)}`
- **Features:** `{results.get('n_features', 0)}`
- **Classes:** `{results.get('n_classes', 0)}`

---

## 💡 INSIGHTS

{results.get('insights', '- System evolved optimal architecture through neuroevolution\\n- Performance metrics indicate strong generalization\\n- Model ready for deployment and inference')}

---

**Generated by VinoGen-CyberCore v1.0.0**
*The Neural Architecture Search Engine*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return str(output_path)


def ensure_directories(base_dir: Path):
    """Ensure all required directories exist."""
    from pathlib import Path
    base_dir = Path(base_dir)  # Convert to Path if string
    directories = [
        base_dir / "input",
        base_dir / "output",
        base_dir / "output" / "models",
        base_dir / "assets",
        base_dir / "config"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
