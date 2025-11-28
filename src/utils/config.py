"""
VinoGen-CyberCore: Configuration Settings
"""

import yaml
from pathlib import Path
from typing import Dict


class Config:
    """Central configuration for the entire system."""
    
    # Project metadata
    PROJECT_NAME = "VinoGen-CyberCore"
    VERSION = "1.0.0"
    AUTHOR = "Neuroevolution AI Lab"
    
    # Paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"
    OUTPUT_DIR = ROOT_DIR / "output"
    ASSETS_DIR = ROOT_DIR / "assets"
    CONFIG_DIR = ROOT_DIR / "config"
    
    # Data settings
    DATASET_PATH = DATA_DIR / "winequality.csv"
    TASK = "classification"  # or "regression"
    TEST_SIZE = 0.2
    VAL_SIZE = 0.25
    RANDOM_SEED = 42
    
    # Genetic Algorithm parameters
    GA_POPULATION_SIZE = 20
    GA_GENERATIONS = 10
    GA_MUTATION_RATE = 0.3
    GA_CROSSOVER_RATE = 0.7
    GA_ELITISM_RATIO = 0.2
    
    # Network architecture constraints
    MAX_LAYERS = 5
    MAX_NEURONS = 256
    MIN_NEURONS = 16
    AVAILABLE_ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu']
    LEARNING_RATE_RANGE = (0.0001, 0.01)
    
    # Training parameters
    EPOCHS_PER_GENOME = 50
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 10
    
    # Visualization settings
    SAVE_TOPOLOGY = True
    SAVE_ANIMATION = True
    SAVE_LEARNING_CURVES = True
    SAVE_CONFUSION_MATRIX = True
    SAVE_3D_LANDSCAPE = True
    ANIMATION_FPS = 10
    
    # UI settings
    VERBOSE = True
    SHOW_MATH_EXPLANATIONS = True
    
    @classmethod
    def save_to_yaml(cls, filepath: Path = None):
        """Save configuration to YAML file."""
        if filepath is None:
            filepath = cls.CONFIG_DIR / "config.yaml"
        
        filepath.parent.mkdir(exist_ok=True, parents=True)
        
        config_dict = {
            'project': {
                'name': cls.PROJECT_NAME,
                'version': cls.VERSION,
                'author': cls.AUTHOR
            },
            'data': {
                'dataset_path': str(cls.DATASET_PATH),
                'task': cls.TASK,
                'test_size': cls.TEST_SIZE,
                'val_size': cls.VAL_SIZE,
                'random_seed': cls.RANDOM_SEED
            },
            'genetic_algorithm': {
                'population_size': cls.GA_POPULATION_SIZE,
                'generations': cls.GA_GENERATIONS,
                'mutation_rate': cls.GA_MUTATION_RATE,
                'crossover_rate': cls.GA_CROSSOVER_RATE,
                'elitism_ratio': cls.GA_ELITISM_RATIO
            },
            'architecture': {
                'max_layers': cls.MAX_LAYERS,
                'max_neurons': cls.MAX_NEURONS,
                'min_neurons': cls.MIN_NEURONS,
                'activations': cls.AVAILABLE_ACTIVATIONS,
                'learning_rate_range': list(cls.LEARNING_RATE_RANGE)
            },
            'training': {
                'epochs_per_genome': cls.EPOCHS_PER_GENOME,
                'batch_size': cls.BATCH_SIZE,
                'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE
            },
            'visualization': {
                'save_topology': cls.SAVE_TOPOLOGY,
                'save_animation': cls.SAVE_ANIMATION,
                'save_learning_curves': cls.SAVE_LEARNING_CURVES,
                'save_confusion_matrix': cls.SAVE_CONFUSION_MATRIX,
                'save_3d_landscape': cls.SAVE_3D_LANDSCAPE,
                'animation_fps': cls.ANIMATION_FPS
            },
            'ui': {
                'verbose': cls.VERBOSE,
                'show_math_explanations': cls.SHOW_MATH_EXPLANATIONS
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        return filepath
    
    @classmethod
    def load_from_yaml(cls, filepath: Path = None):
        """Load configuration from YAML file."""
        if filepath is None:
            filepath = cls.CONFIG_DIR / "config.yaml"
        
        if not filepath.exists():
            return
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update class attributes
        if 'data' in config_dict:
            cls.TASK = config_dict['data'].get('task', cls.TASK)
            cls.TEST_SIZE = config_dict['data'].get('test_size', cls.TEST_SIZE)
        
        if 'genetic_algorithm' in config_dict:
            ga = config_dict['genetic_algorithm']
            cls.GA_POPULATION_SIZE = ga.get('population_size', cls.GA_POPULATION_SIZE)
            cls.GA_GENERATIONS = ga.get('generations', cls.GA_GENERATIONS)
        
        # Add more loading logic as needed


# Mathematical equations for educational display
MATH_EQUATIONS = {
    'forward_pass': {
        'name': 'Forward Propagation',
        'equation': 'z[l] = W[l] @ a[l-1] + b[l]\na[l] = σ(z[l])',
        'description': (
            'Forward pass computes the output of each layer by:\n'
            '  1. Linear transformation: z = Wx + b\n'
            '  2. Apply activation function: a = σ(z)\n'
            'Where W is weight matrix, b is bias, σ is activation function'
        )
    },
    'cross_entropy': {
        'name': 'Cross-Entropy Loss (Classification)',
        'equation': 'L = -Σ y_true * log(y_pred)',
        'description': (
            'Cross-entropy measures the difference between predicted and true distributions.\n'
            'Lower values indicate better predictions.\n'
            'Commonly used for multi-class classification problems.'
        )
    },
    'mse': {
        'name': 'Mean Squared Error (Regression)',
        'equation': 'MSE = (1/n) * Σ(y_true - y_pred)²',
        'description': (
            'MSE calculates the average squared difference between predictions and targets.\n'
            'Penalizes larger errors more heavily.\n'
            'Standard loss function for regression tasks.'
        )
    },
    'backpropagation': {
        'name': 'Backpropagation',
        'equation': '∂L/∂W[l] = ∂L/∂a[l] * ∂a[l]/∂z[l] * ∂z[l]/∂W[l]',
        'description': (
            'Backpropagation computes gradients using the chain rule:\n'
            '  1. Calculate loss gradient\n'
            '  2. Propagate backwards through layers\n'
            '  3. Compute weight gradients for optimization'
        )
    },
    'gradient_descent': {
        'name': 'Gradient Descent Update',
        'equation': 'W ← W - α * ∂L/∂W',
        'description': (
            'Gradient descent updates weights to minimize loss:\n'
            '  α (learning rate) controls step size\n'
            '  Negative gradient points toward loss minimum\n'
            '  Iteratively improves network performance'
        )
    },
    'fitness': {
        'name': 'Genetic Algorithm Fitness',
        'equation': 'F(genome) = α * accuracy + β * (1/complexity)',
        'description': (
            'Fitness function balances performance and efficiency:\n'
            '  - Higher accuracy = better predictions\n'
            '  - Lower complexity = simpler, faster networks\n'
            '  - α and β control the tradeoff'
        )
    }
}
