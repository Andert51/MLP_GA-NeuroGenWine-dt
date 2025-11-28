"""
VinoGen-CyberCore: Main Orchestrator
The ultimate neuroevolution system for wine quality prediction.

This system combines:
- Genetic Algorithms for neural architecture search
- PyTorch-based MLP with dynamic topology
- Cyberpunk-themed CLI interface
- Comprehensive visualizations

Author: Neuroevolution AI Lab
Version: 1.0.0
"""

import sys
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Add src to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

# Import modules
from src.data import DataHandler
from src.models import DynamicMLP
from src.genetic import GeneticOptimizer, Genome
from src.visualization import Visualizer
from src.ui import CyberpunkUI
from src.utils import (
    Config, MATH_EQUATIONS, set_random_seeds, 
    EarlyStopping, save_json, get_device
)


class VinoGenCyberCore:
    """
    Main orchestrator for the VinoGen-CyberCore system.
    
    This class manages the entire pipeline:
    1. Data loading and preprocessing
    2. Genetic algorithm evolution
    3. Model training and evaluation
    4. Visualization generation
    5. UI management
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the VinoGen-CyberCore system.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.ui = CyberpunkUI()
        self.visualizer = Visualizer(output_dir=str(self.config.OUTPUT_DIR))
        self.device = get_device()
        
        # Components
        self.data_handler = None
        self.genetic_optimizer = None
        self.best_model = None
        self.best_genome = None
        
        # Results storage
        self.results = {}
        
        # Set random seeds
        set_random_seeds(self.config.RANDOM_SEED)
    
    def run(self):
        """Execute the complete pipeline."""
        try:
            # Boot sequence
            self.ui.show_boot_sequence()
            time.sleep(1)
            
            # Phase 1: Data Loading
            self.ui.show_header("PHASE 1: DATA MATRIX INITIALIZATION", 
                               "Loading and preprocessing wine dataset...")
            self.load_data()
            
            # Phase 2: Mathematical Foundations
            if self.config.SHOW_MATH_EXPLANATIONS:
                self.show_math_explanations()
            
            # Phase 3: Genetic Evolution
            self.ui.show_header("PHASE 3: GENETIC EVOLUTION", 
                               "Evolving optimal neural architecture...")
            self.evolve_architecture()
            
            # Phase 4: Final Training
            self.ui.show_header("PHASE 4: FINAL TRAINING", 
                               "Training the evolved champion network...")
            self.train_best_model()
            
            # Phase 5: Evaluation
            self.ui.show_header("PHASE 5: EVALUATION", 
                               "Testing on holdout dataset...")
            self.evaluate_model()
            
            # Phase 6: Visualization
            self.ui.show_header("PHASE 6: VISUALIZATION ENGINE", 
                               "Generating stunning visuals...")
            self.generate_visualizations()
            
            # Phase 7: Results
            self.ui.show_header("PHASE 7: MISSION REPORT", 
                               "Compiling final results...")
            self.display_results()
            
            # Completion
            self.ui.show_completion_banner()
            self.ui.log("All output files saved to: output/", "SUCCESS")
            
        except KeyboardInterrupt:
            self.ui.log("Evolution terminated by user", "WARNING")
            sys.exit(0)
        except Exception as e:
            self.ui.log(f"Critical error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def load_data(self):
        """Load and preprocess the wine dataset."""
        self.ui.log("Initializing data handler...", "SYSTEM")
        
        # Create data handler
        data_path = self.config.DATASET_PATH if self.config.DATASET_PATH.exists() else None
        self.data_handler = DataHandler(
            data_path=str(data_path) if data_path else None,
            task=self.config.TASK
        )
        
        # Load data
        self.ui.log("Loading dataset...", "INFO")
        success = self.data_handler.load_data()
        
        if not success:
            self.ui.log("Data loading failed!", "ERROR")
            sys.exit(1)
        
        # Display info
        data_info = self.data_handler.get_info()
        self.ui.show_data_info(data_info)
        
        self.ui.log(f"Dataset loaded: {data_info['samples']} samples, "
                   f"{data_info['features']} features", "SUCCESS")
        
        # Store in results
        self.results['dataset_info'] = data_info
    
    def show_math_explanations(self):
        """Display mathematical foundations."""
        self.ui.show_header("PHASE 2: MATHEMATICAL FOUNDATIONS", 
                           "Understanding the neural network mathematics...")
        
        # Show key equations
        equations_to_show = ['forward_pass', 'cross_entropy' if self.config.TASK == 'classification' else 'mse',
                            'backpropagation', 'gradient_descent', 'fitness']
        
        for eq_key in equations_to_show:
            if eq_key in MATH_EQUATIONS:
                eq = MATH_EQUATIONS[eq_key]
                self.ui.show_math_explanation(
                    eq['name'],
                    eq['equation'],
                    eq['description']
                )
                time.sleep(1)
    
    def evolve_architecture(self):
        """Run genetic algorithm to evolve optimal architecture."""
        self.ui.log("Initializing Genetic Algorithm Engine...", "SYSTEM")
        
        # Create genetic optimizer
        self.genetic_optimizer = GeneticOptimizer(
            population_size=self.config.GA_POPULATION_SIZE,
            generations=self.config.GA_GENERATIONS,
            mutation_rate=self.config.GA_MUTATION_RATE,
            crossover_rate=self.config.GA_CROSSOVER_RATE,
            elitism_ratio=self.config.GA_ELITISM_RATIO,
            max_layers=self.config.MAX_LAYERS,
            max_neurons=self.config.MAX_NEURONS,
            min_neurons=self.config.MIN_NEURONS,
            available_activations=self.config.AVAILABLE_ACTIVATIONS,
            lr_range=self.config.LEARNING_RATE_RANGE
        )
        
        # Initialize population
        self.ui.log("Creating initial population...", "INFO")
        population = self.genetic_optimizer.initialize_population()
        self.ui.log(f"Population created: {len(population)} genomes", "SUCCESS")
        
        # Evolution loop
        for generation in range(self.config.GA_GENERATIONS):
            self.ui.log(f"\n{'='*70}", "SYSTEM")
            self.ui.log(f"GENERATION {generation + 1}/{self.config.GA_GENERATIONS}", "SYSTEM")
            self.ui.log(f"{'='*70}\n", "SYSTEM")
            
            # Evaluate fitness for all genomes
            with self.ui.create_progress_bar(
                f"Evaluating Generation {generation + 1}", 
                total=len(population)
            ) as progress:
                task = progress.add_task("Training genomes...", total=len(population))
                
                for i, genome in enumerate(population):
                    # Create fitness function
                    fitness = self.evaluate_genome_fitness(genome)
                    genome.fitness = fitness
                    
                    progress.update(task, advance=1, 
                                  description=f"Genome {i+1}/{len(population)} | Fitness: {fitness:.4f}")
            
            # Show top performers
            self.ui.show_genome_table(population, top_n=5)
            
            # Evolution statistics
            fitnesses = [g.fitness for g in population]
            self.ui.show_evolution_progress(
                generation + 1,
                self.config.GA_GENERATIONS,
                max(fitnesses),
                np.mean(fitnesses),
                np.std([len(g.hidden_layers) for g in population])
            )
            
            # Evolve to next generation (except last one)
            if generation < self.config.GA_GENERATIONS - 1:
                self.ui.log("Applying selection, crossover, and mutation...", "INFO")
                population = self.genetic_optimizer.evolve_generation(
                    fitness_function=self.evaluate_genome_fitness,
                    generation=generation
                )
                self.ui.log("New generation created", "SUCCESS")
            
            time.sleep(0.5)
        
        # Get best genome
        self.best_genome = self.genetic_optimizer.get_best_genome()
        self.ui.log(f"\nBest genome found with fitness: {self.best_genome.fitness:.6f}", "SUCCESS")
        self.ui.log(f"Architecture: {' → '.join([str(n) for n in self.best_genome.hidden_layers])}", "INFO")
        
        # Store in results
        self.results['best_genome'] = self.best_genome.to_dict()
        self.results['best_genome']['fitness'] = self.best_genome.fitness
        self.results['best_genome']['generation'] = self.best_genome.generation
        self.results['evolution_history'] = self.genetic_optimizer.get_statistics()['history']
    
    def evaluate_genome_fitness(self, genome: Genome) -> float:
        """
        Evaluate a genome's fitness by training and validating.
        
        Args:
            genome: Genome to evaluate
            
        Returns:
            Fitness score (validation accuracy or R²)
        """
        try:
            # Get data
            data_splits = self.data_handler.get_data_splits()
            
            # Create model
            model = DynamicMLP(
                genome=genome.to_dict(),
                input_dim=self.data_handler.n_features,
                output_dim=self.data_handler.n_classes if self.config.TASK == 'classification' else 1,
                task=self.config.TASK
            )
            
            # Training loop with early stopping
            early_stopping = EarlyStopping(patience=self.config.EARLY_STOPPING_PATIENCE)
            
            for epoch in range(self.config.EPOCHS_PER_GENOME):
                # Train
                train_loss = model.train_epoch(
                    data_splits['X_train'],
                    data_splits['y_train'],
                    batch_size=self.config.BATCH_SIZE
                )
                
                # Validate
                val_metrics = model.evaluate(
                    data_splits['X_val'],
                    data_splits['y_val']
                )
                
                # Early stopping check
                if early_stopping(val_metrics['loss']):
                    break
            
            # Calculate fitness
            if self.config.TASK == 'classification':
                fitness = val_metrics['accuracy']
            else:
                fitness = val_metrics['r2']
            
            return fitness
            
        except Exception as e:
            self.ui.log(f"Genome evaluation failed: {e}", "WARNING")
            return 0.0
    
    def train_best_model(self):
        """Train the best evolved model with full monitoring."""
        self.ui.log("Creating champion network...", "SYSTEM")
        
        # Create model with best genome
        data_splits = self.data_handler.get_data_splits()
        self.best_model = DynamicMLP(
            genome=self.best_genome.to_dict(),
            input_dim=self.data_handler.n_features,
            output_dim=self.data_handler.n_classes if self.config.TASK == 'classification' else 1,
            task=self.config.TASK
        )
        
        # Show architecture
        arch_summary = self.best_model.get_architecture_summary()
        self.ui.show_architecture_summary(arch_summary)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Training loop
        self.ui.log("Beginning training...", "INFO")
        
        with self.ui.create_progress_bar("Training Progress", 
                                        total=self.config.EPOCHS_PER_GENOME * 2) as progress:
            task = progress.add_task("Epochs...", total=self.config.EPOCHS_PER_GENOME * 2)
            
            for epoch in range(self.config.EPOCHS_PER_GENOME * 2):  # Train longer for final model
                # Train
                train_loss = self.best_model.train_epoch(
                    data_splits['X_train'],
                    data_splits['y_train'],
                    batch_size=self.config.BATCH_SIZE
                )
                
                # Validate
                train_metrics = self.best_model.evaluate(
                    data_splits['X_train'],
                    data_splits['y_train']
                )
                val_metrics = self.best_model.evaluate(
                    data_splits['X_val'],
                    data_splits['y_val']
                )
                
                # Store history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_metrics['loss'])
                
                if self.config.TASK == 'classification':
                    history['train_acc'].append(train_metrics['accuracy'])
                    history['val_acc'].append(val_metrics['accuracy'])
                
                # Update progress
                progress.update(task, advance=1,
                              description=f"Epoch {epoch+1} | Loss: {val_metrics['loss']:.4f}")
                
                # Periodic display
                if (epoch + 1) % 10 == 0:
                    self.ui.show_training_progress(
                        epoch + 1,
                        self.config.EPOCHS_PER_GENOME * 2,
                        train_loss,
                        val_metrics['loss'],
                        train_metrics.get('accuracy'),
                        val_metrics.get('accuracy')
                    )
        
        self.ui.log("Training completed!", "SUCCESS")
        
        # Store history
        self.results['training_history'] = history
    
    def evaluate_model(self):
        """Evaluate the final model on test set."""
        data_splits = self.data_handler.get_data_splits()
        
        self.ui.log("Running final evaluation...", "INFO")
        
        # Evaluate
        test_metrics = self.best_model.evaluate(
            data_splits['X_test'],
            data_splits['y_test']
        )
        
        # Get predictions
        y_pred = self.best_model.predict(data_splits['X_test'])
        y_true = data_splits['y_test']
        
        # Display metrics
        self.ui.log(f"\nTest Set Performance:", "SYSTEM")
        for metric, value in test_metrics.items():
            self.ui.log(f"  {metric.upper()}: {value:.6f}", "INFO")
        
        # Classification report
        if self.config.TASK == 'classification':
            report = classification_report(y_true, y_pred)
            self.ui.log("\nClassification Report:", "SYSTEM")
            print(report)
            self.results['classification_report'] = report
        
        # Store results
        self.results['test_metrics'] = test_metrics
        self.results['predictions'] = {
            'y_true': y_true.tolist() if hasattr(y_true, 'tolist') else y_true,
            'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
        }
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        data_splits = self.data_handler.get_data_splits()
        
        viz_tasks = []
        
        # Network topology
        if self.config.SAVE_TOPOLOGY:
            viz_tasks.append(("Network Topology", lambda: self.visualizer.plot_network_topology(
                self.best_genome.to_dict(),
                self.data_handler.n_features,
                self.data_handler.n_classes if self.config.TASK == 'classification' else 1
            )))
        
        # Activation flow animation
        if self.config.SAVE_ANIMATION:
            sample_input, _ = self.data_handler.get_sample_for_visualization()
            if sample_input is not None:
                viz_tasks.append(("Activation Flow", lambda: self.visualizer.create_activation_flow_animation(
                    self.best_model,
                    sample_input,
                    self.best_genome.to_dict(),
                    fps=self.config.ANIMATION_FPS
                )))
        
        # Learning curves
        if self.config.SAVE_LEARNING_CURVES and 'training_history' in self.results:
            viz_tasks.append(("Learning Curves", lambda: self.visualizer.plot_learning_curves(
                self.results['training_history']
            )))
        
        # Confusion matrix
        if self.config.SAVE_CONFUSION_MATRIX and self.config.TASK == 'classification':
            viz_tasks.append(("Confusion Matrix", lambda: self.visualizer.plot_confusion_matrix(
                data_splits['y_test'],
                self.best_model.predict(data_splits['X_test'])
            )))
        
        # Evolution history
        if 'evolution_history' in self.results:
            viz_tasks.append(("Evolution History", lambda: self.visualizer.plot_evolution_history(
                self.results['evolution_history']
            )))
        
        # 3D landscape
        if self.config.SAVE_3D_LANDSCAPE and 'evolution_history' in self.results:
            viz_tasks.append(("3D Loss Landscape", lambda: self.visualizer.plot_3d_loss_landscape(
                self.results['evolution_history']
            )))
        
        # Generate all
        with self.ui.create_progress_bar("Generating Visualizations", 
                                        total=len(viz_tasks)) as progress:
            task = progress.add_task("Creating visuals...", total=len(viz_tasks))
            
            for name, viz_func in viz_tasks:
                try:
                    filepath = viz_func()
                    progress.update(task, advance=1, description=f"Generated {name}")
                    self.ui.log(f"Created: {filepath}", "SUCCESS")
                except Exception as e:
                    self.ui.log(f"Failed to create {name}: {e}", "WARNING")
                    progress.update(task, advance=1)
        
        # Generate report
        self.visualizer.generate_report(self.results)
        self.ui.log("Generated: output/final_report.txt", "SUCCESS")
    
    def display_results(self):
        """Display final results dashboard."""
        self.ui.show_results_dashboard(self.results)
        
        # Save results to JSON
        results_path = self.config.OUTPUT_DIR / "results.json"
        save_json(self.results, results_path)
        self.ui.log(f"Results saved to: {results_path}", "SUCCESS")


def main():
    """Main entry point."""
    # Create config
    config = Config()
    
    # Save default config
    config.save_to_yaml()
    
    # Create and run system
    system = VinoGenCyberCore(config)
    system.run()


if __name__ == "__main__":
    main()
