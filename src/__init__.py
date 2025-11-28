"""
VinoGen-CyberCore: Main Package Initialization
"""

__version__ = "1.0.0"
__author__ = "Neuroevolution AI Lab"
__description__ = "Hybrid Neural Network + Genetic Algorithm System for Wine Quality Prediction"

# Import main components
from .data import DataHandler
from .models import DynamicMLP
from .genetic import GeneticOptimizer, Genome
from .visualization import Visualizer
from .ui import CyberpunkUI
from .utils import Config

__all__ = [
    'DataHandler',
    'DynamicMLP',
    'GeneticOptimizer',
    'Genome',
    'Visualizer',
    'CyberpunkUI',
    'Config'
]
