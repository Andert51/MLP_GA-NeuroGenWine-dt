"""
VinoGen-CyberCore: Unit Tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data import DataHandler
from src.models import DynamicMLP
from src.genetic import GeneticOptimizer, Genome
from src.utils import Config, set_random_seeds


class TestDataHandler:
    """Test data handling functionality."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        set_random_seeds(42)
        handler = DataHandler(task="classification")
        success = handler.load_data()
        
        assert success
        assert handler.X_train is not None
        assert handler.X_test is not None
        assert handler.n_features == 11
        assert handler.n_classes > 0
    
    def test_data_splits(self):
        """Test data splitting."""
        handler = DataHandler(task="classification")
        handler.load_data()
        
        splits = handler.get_data_splits()
        
        assert 'X_train' in splits
        assert 'X_val' in splits
        assert 'X_test' in splits
        assert len(splits['X_train']) > 0


class TestDynamicMLP:
    """Test neural network functionality."""
    
    def test_network_creation(self):
        """Test network instantiation."""
        genome = {
            'hidden_layers': [64, 32],
            'activation_functions': ['relu', 'tanh'],
            'learning_rate': 0.001
        }
        
        model = DynamicMLP(genome, input_dim=11, output_dim=7, task="classification")
        
        assert model is not None
        assert len(model.layers) == 2
        assert model.learning_rate == 0.001
    
    def test_forward_pass(self):
        """Test forward propagation."""
        genome = {
            'hidden_layers': [64, 32],
            'activation_functions': ['relu', 'tanh'],
            'learning_rate': 0.001
        }
        
        model = DynamicMLP(genome, input_dim=11, output_dim=7, task="classification")
        
        import torch
        X = torch.randn(10, 11)
        output = model.forward(X)
        
        assert output.shape == (10, 7)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        genome = {
            'hidden_layers': [64, 32],
            'activation_functions': ['relu', 'tanh'],
            'learning_rate': 0.001
        }
        
        model = DynamicMLP(genome, input_dim=11, output_dim=7, task="classification")
        param_count = model.count_parameters()
        
        # Expected: (11*64 + 64) + (64*32 + 32) + (32*7 + 7)
        expected = (11*64 + 64) + (64*32 + 32) + (32*7 + 7)
        assert param_count == expected


class TestGeneticOptimizer:
    """Test genetic algorithm functionality."""
    
    def test_population_initialization(self):
        """Test population creation."""
        ga = GeneticOptimizer(population_size=10, generations=5)
        population = ga.initialize_population()
        
        assert len(population) == 10
        assert all(isinstance(g, Genome) for g in population)
    
    def test_genome_structure(self):
        """Test genome attributes."""
        ga = GeneticOptimizer()
        population = ga.initialize_population()
        genome = population[0]
        
        assert hasattr(genome, 'hidden_layers')
        assert hasattr(genome, 'activation_functions')
        assert hasattr(genome, 'learning_rate')
        assert len(genome.hidden_layers) > 0
        assert len(genome.activation_functions) == len(genome.hidden_layers)
    
    def test_mutation(self):
        """Test mutation operator."""
        ga = GeneticOptimizer(mutation_rate=1.0)  # High mutation for testing
        
        genome = Genome(
            hidden_layers=[64, 32],
            activation_functions=['relu', 'tanh'],
            learning_rate=0.001
        )
        
        mutated = ga.mutate(genome)
        
        # Should be different (with high probability)
        assert mutated is not genome  # Different object
    
    def test_crossover(self):
        """Test crossover operator."""
        ga = GeneticOptimizer(crossover_rate=1.0)
        
        parent1 = Genome(
            hidden_layers=[64, 32],
            activation_functions=['relu', 'tanh'],
            learning_rate=0.001
        )
        
        parent2 = Genome(
            hidden_layers=[128, 64, 32],
            activation_functions=['sigmoid', 'relu', 'tanh'],
            learning_rate=0.002
        )
        
        child1, child2 = ga.crossover(parent1, parent2)
        
        assert child1 is not parent1
        assert child2 is not parent2


class TestConfig:
    """Test configuration management."""
    
    def test_config_attributes(self):
        """Test configuration has required attributes."""
        config = Config()
        
        assert hasattr(config, 'PROJECT_NAME')
        assert hasattr(config, 'GA_POPULATION_SIZE')
        assert hasattr(config, 'EPOCHS_PER_GENOME')
        assert config.TASK in ['classification', 'regression']
    
    def test_config_yaml_save(self, tmp_path):
        """Test saving configuration to YAML."""
        config = Config()
        config.CONFIG_DIR = tmp_path
        
        filepath = config.save_to_yaml()
        
        assert Path(filepath).exists()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
