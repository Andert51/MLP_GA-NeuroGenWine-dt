"""
VinoGen-CyberCore: Genetic Algorithm Optimizer
Evolves optimal neural network architectures through natural selection.
"""

import numpy as np
import copy
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import random


@dataclass
class Genome:
    """
    Represents a neural network architecture as a genetic sequence.
    
    Genes:
    - hidden_layers: List of neuron counts per layer
    - activation_functions: List of activation function names
    - learning_rate: Optimizer learning rate
    - fitness: Performance score (higher is better)
    """
    hidden_layers: List[int]
    activation_functions: List[str]
    learning_rate: float
    fitness: float = 0.0
    generation: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for model initialization."""
        return {
            'hidden_layers': self.hidden_layers,
            'activation_functions': self.activation_functions,
            'learning_rate': self.learning_rate
        }
    
    def __str__(self) -> str:
        """String representation."""
        layers_str = " → ".join([str(n) for n in self.hidden_layers])
        acts_str = ", ".join(self.activation_functions)
        return f"[{layers_str}] | Acts: {acts_str} | LR: {self.learning_rate:.6f} | Fitness: {self.fitness:.4f}"


class GeneticOptimizer:
    """
    Genetic Algorithm for Neural Architecture Search (NAS).
    
    Evolutionary Process:
    =====================
    
    1. INITIALIZATION
       - Create random population of network architectures
    
    2. EVALUATION
       - Train each network on dataset
       - Compute fitness (validation accuracy/R² score)
    
    3. SELECTION
       - Select best individuals (Tournament/Roulette)
       - Keep elites (top performers)
    
    4. CROSSOVER (Mating)
       - Combine two parent genomes
       - Create offspring with mixed traits
    
    5. MUTATION
       - Random changes to architecture
       - Add/remove neurons, change activations
    
    6. REPEAT
       - Evolve over multiple generations
    
    Mathematical Foundation:
    ------------------------
    
    Fitness Function:
        F(genome) = α * accuracy + β * (1/complexity)
        
    Where complexity = total_parameters
    
    Selection Probability (Roulette Wheel):
        P(i) = fitness(i) / Σ fitness(j)
    
    Mutation Rate:
        Adaptive: μ = μ_base * e^(-generation/decay)
    """
    
    def __init__(self,
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 elitism_ratio: float = 0.2,
                 max_layers: int = 5,
                 max_neurons: int = 256,
                 min_neurons: int = 16,
                 available_activations: Optional[List[str]] = None,
                 lr_range: Tuple[float, float] = (0.0001, 0.01)):
        """
        Initialize the Genetic Optimizer.
        
        Args:
            population_size: Number of individuals per generation
            generations: Number of evolutionary cycles
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_ratio: Fraction of top performers to preserve
            max_layers: Maximum hidden layers
            max_neurons: Maximum neurons per layer
            min_neurons: Minimum neurons per layer
            available_activations: List of activation functions to use
            lr_range: Learning rate range (min, max)
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        
        # Architecture constraints
        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.min_neurons = min_neurons
        self.lr_range = lr_range
        
        # Available activation functions
        self.available_activations = available_activations or [
            'relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu'
        ]
        
        # Population
        self.population: List[Genome] = []
        self.best_genome: Optional[Genome] = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }
        
        # Elite count
        self.elite_count = max(1, int(population_size * elitism_ratio))
    
    def initialize_population(self) -> List[Genome]:
        """
        Create initial random population of network architectures.
        
        Returns:
            List of random genomes
        """
        population = []
        
        for _ in range(self.population_size):
            # Random number of hidden layers
            n_layers = random.randint(1, self.max_layers)
            
            # Random neurons per layer
            hidden_layers = [
                random.randint(self.min_neurons, self.max_neurons)
                for _ in range(n_layers)
            ]
            
            # Random activation functions
            activation_functions = [
                random.choice(self.available_activations)
                for _ in range(n_layers)
            ]
            
            # Random learning rate
            learning_rate = random.uniform(*self.lr_range)
            
            genome = Genome(
                hidden_layers=hidden_layers,
                activation_functions=activation_functions,
                learning_rate=learning_rate
            )
            
            population.append(genome)
        
        self.population = population
        return population
    
    def evaluate_fitness(self, 
                        genome: Genome, 
                        fitness_function: Callable) -> float:
        """
        Evaluate a genome's fitness using the provided function.
        
        Args:
            genome: Genome to evaluate
            fitness_function: Function that trains and evaluates the network
            
        Returns:
            Fitness score
        """
        fitness = fitness_function(genome)
        genome.fitness = fitness
        return fitness
    
    def selection(self, tournament_size: int = 3) -> Genome:
        """
        Tournament selection: Pick best from random subset.
        
        Args:
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected genome
        """
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness)
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """
        Single-point crossover: Combine two parent genomes.
        
        Creates offspring by mixing architectural traits from both parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring genomes
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copies
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # Choose crossover point
        min_len = min(len(parent1.hidden_layers), len(parent2.hidden_layers))
        
        if min_len > 1:
            crossover_point = random.randint(1, min_len - 1)
            
            # Create offspring
            child1_layers = parent1.hidden_layers[:crossover_point] + parent2.hidden_layers[crossover_point:]
            child1_acts = parent1.activation_functions[:crossover_point] + parent2.activation_functions[crossover_point:]
            
            child2_layers = parent2.hidden_layers[:crossover_point] + parent1.hidden_layers[crossover_point:]
            child2_acts = parent2.activation_functions[:crossover_point] + parent1.activation_functions[crossover_point:]
            
            # Inherit learning rates (average)
            child1_lr = (parent1.learning_rate + parent2.learning_rate) / 2
            child2_lr = child1_lr
        else:
            # Can't do meaningful crossover, return modified copies
            child1_layers = parent1.hidden_layers[:]
            child1_acts = parent1.activation_functions[:]
            child1_lr = parent1.learning_rate
            
            child2_layers = parent2.hidden_layers[:]
            child2_acts = parent2.activation_functions[:]
            child2_lr = parent2.learning_rate
        
        child1 = Genome(
            hidden_layers=child1_layers,
            activation_functions=child1_acts,
            learning_rate=child1_lr
        )
        
        child2 = Genome(
            hidden_layers=child2_layers,
            activation_functions=child2_acts,
            learning_rate=child2_lr
        )
        
        return child1, child2
    
    def mutate(self, genome: Genome) -> Genome:
        """
        Apply random mutations to a genome.
        
        Mutation Types:
        - Add/remove neurons from a layer
        - Add/remove entire layer
        - Change activation function
        - Adjust learning rate
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        mutated = copy.deepcopy(genome)
        
        # Neuron count mutation
        if random.random() < self.mutation_rate:
            layer_idx = random.randint(0, len(mutated.hidden_layers) - 1)
            change = random.randint(-32, 32)
            mutated.hidden_layers[layer_idx] = np.clip(
                mutated.hidden_layers[layer_idx] + change,
                self.min_neurons,
                self.max_neurons
            )
        
        # Add layer mutation
        if random.random() < self.mutation_rate * 0.5 and len(mutated.hidden_layers) < self.max_layers:
            new_neurons = random.randint(self.min_neurons, self.max_neurons)
            new_activation = random.choice(self.available_activations)
            insert_pos = random.randint(0, len(mutated.hidden_layers))
            mutated.hidden_layers.insert(insert_pos, new_neurons)
            mutated.activation_functions.insert(insert_pos, new_activation)
        
        # Remove layer mutation
        if random.random() < self.mutation_rate * 0.5 and len(mutated.hidden_layers) > 1:
            remove_idx = random.randint(0, len(mutated.hidden_layers) - 1)
            del mutated.hidden_layers[remove_idx]
            del mutated.activation_functions[remove_idx]
        
        # Activation function mutation
        if random.random() < self.mutation_rate:
            layer_idx = random.randint(0, len(mutated.activation_functions) - 1)
            mutated.activation_functions[layer_idx] = random.choice(self.available_activations)
        
        # Learning rate mutation
        if random.random() < self.mutation_rate:
            lr_change = random.uniform(-0.001, 0.001)
            mutated.learning_rate = np.clip(
                mutated.learning_rate + lr_change,
                *self.lr_range
            )
        
        return mutated
    
    def evolve_generation(self, fitness_function: Callable, generation: int) -> List[Genome]:
        """
        Evolve one generation through selection, crossover, and mutation.
        
        Args:
            fitness_function: Function to evaluate fitness
            generation: Current generation number
            
        Returns:
            New population
        """
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Track best
        if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
            self.best_genome = copy.deepcopy(self.population[0])
            self.best_genome.generation = generation
        
        # Statistics
        fitnesses = [g.fitness for g in self.population]
        self.history['best_fitness'].append(max(fitnesses))
        self.history['avg_fitness'].append(np.mean(fitnesses))
        
        # Calculate diversity (std of layer counts)
        layer_counts = [len(g.hidden_layers) for g in self.population]
        self.history['diversity'].append(np.std(layer_counts))
        
        # Store full population data for animations
        if 'population_data' not in self.history:
            self.history['population_data'] = []
        
        self.history['population_data'].append({
            'generation': generation,
            'fitness_scores': fitnesses.copy(),
            'architectures': [{'hidden_layers': g.hidden_layers.copy(), 
                              'activations': g.activation_functions.copy(),
                              'lr': g.learning_rate} for g in self.population]
        })
        
        # Elitism: Keep top performers
        new_population = self.population[:self.elite_count]
        
        # Generate rest through evolution
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.selection()
            parent2 = self.selection()
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add to new population
            child1.generation = generation + 1
            child2.generation = generation + 1
            new_population.extend([child1, child2])
        
        # Trim to population size
        self.population = new_population[:self.population_size]
        
        return self.population
    
    def get_best_genome(self) -> Optional[Genome]:
        """Return the best genome found."""
        return self.best_genome
    
    def get_statistics(self) -> Dict:
        """Get evolution statistics."""
        return {
            'history': self.history,
            'best_genome': self.best_genome,
            'population_size': self.population_size,
            'generations': self.generations,
            'best_fitness': self.best_genome.fitness if self.best_genome else 0.0
        }
    
    def print_population_summary(self, generation: int, top_n: int = 5):
        """Print summary of current population."""
        print(f"\n{'═' * 70}")
        print(f"  GENERATION {generation} SUMMARY")
        print(f"{'═' * 70}")
        
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        
        print(f"\n🏆 TOP {top_n} ARCHITECTURES:")
        print(f"{'─' * 70}")
        
        for i, genome in enumerate(sorted_pop[:top_n], 1):
            print(f"\n  Rank {i}:")
            print(f"    Architecture: {' → '.join([str(n) for n in genome.hidden_layers])}")
            print(f"    Activations:  {', '.join(genome.activation_functions)}")
            print(f"    Learning Rate: {genome.learning_rate:.6f}")
            print(f"    Fitness:      {genome.fitness:.6f}")
        
        print(f"\n{'─' * 70}")
        print(f"  Best Fitness:    {max(g.fitness for g in self.population):.6f}")
        print(f"  Average Fitness: {np.mean([g.fitness for g in self.population]):.6f}")
        print(f"  Diversity:       {np.std([len(g.hidden_layers) for g in self.population]):.4f}")
        print(f"{'─' * 70}\n")
