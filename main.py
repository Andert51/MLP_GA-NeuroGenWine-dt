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
    EarlyStopping, save_json, get_device,
    save_model, load_model, list_saved_models,
    create_markdown_report, ensure_directories
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
        self.loaded_model = None  # For inference mode
        self.loaded_genome = None
        self.loaded_history = None
        self.loaded_metrics = None
        
        # Results storage
        self.results = {}
        self.history = {}
        
        # Set random seeds
        set_random_seeds(self.config.RANDOM_SEED)
        
        # Ensure directories exist
        ensure_directories(ROOT_DIR)
    
    def run(self):
        """Execute the complete NEW RUN pipeline."""
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
            
            # Phase 7: Results & Saving
            self.ui.show_header("PHASE 7: MISSION REPORT", 
                               "Compiling final results...")
            self.display_results()
            
            # PHASE 3 ENHANCEMENT: Save the trained model
            self.ui.log("Saving trained model...", "SYSTEM")
            model_saved = save_model(
                model=self.best_model,
                genome=self.best_genome,
                history=self.history,
                metrics=self.results.get('test_metrics', {}),
                output_dir=str(self.config.OUTPUT_DIR / 'models')
            )
            
            if model_saved:
                self.ui.log(f"Model saved to: {model_saved}", "SUCCESS")
            
            # PHASE 3 ENHANCEMENT: Generate markdown report
            report_path = self.config.OUTPUT_DIR / 'MISSION_REPORT.md'
            create_markdown_report(
                results=self.results,
                genome=self.best_genome,
                output_path=str(report_path)
            )
            self.ui.log(f"Markdown report saved: {report_path}", "SUCCESS")
            
            # Completion
            self.ui.log("All output files saved to: output/", "SUCCESS")
            self.ui.pause()
            
        except KeyboardInterrupt:
            self.ui.log("Evolution terminated by user", "WARNING")
            self.ui.pause()
        except Exception as e:
            self.ui.log(f"Critical error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            self.ui.pause()
    
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
        self.history = history  # Store in self for model saving
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
        """Generate all visualizations (enhanced for Phase 3)."""
        data_splits = self.data_handler.get_data_splits()
        
        viz_tasks = []
        
        # Network topology
        if self.config.SAVE_TOPOLOGY:
            viz_tasks.append(("Network Topology", lambda: self.visualizer.plot_network_topology(
                self.best_genome.to_dict(),
                self.data_handler.n_features,
                self.data_handler.n_classes if self.config.TASK == 'classification' else 1
            )))
        
        # PHASE 3: Advanced network flow animation
        if self.config.SAVE_ANIMATION:
            sample_input, _ = self.data_handler.get_sample_for_visualization()
            if sample_input is not None:
                # Original activation flow
                viz_tasks.append(("Activation Flow", lambda: self.visualizer.create_activation_flow_animation(
                    self.best_model,
                    sample_input,
                    self.best_genome.to_dict(),
                    fps=self.config.ANIMATION_FPS
                )))
                
                # PHASE 3: Advanced NetworkX-based animation
                viz_tasks.append(("Network Flow Animation", lambda: self.visualizer.animate_network_flow(
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
        
        # PHASE 3: Probability heatmap
        if self.config.TASK == 'classification':
            viz_tasks.append(("Probability Heatmap", lambda: self.visualizer.plot_probability_heatmap(
                self.best_model,
                data_splits['X_test'][:5],  # Top 5 samples
                data_splits['y_test'][:5]
            )))
        
        # PHASE 3: Regression analysis (if regression task)
        if self.config.TASK == 'regression':
            y_pred = self.best_model.predict(data_splits['X_test'])
            viz_tasks.append(("Regression Analysis", lambda: self.visualizer.plot_regression_analysis(
                data_splits['y_test'],
                y_pred
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
                    if filepath:
                        progress.update(task, advance=1, description=f"Generated {name}")
                        self.ui.log(f"Created: {filepath}", "SUCCESS")
                    else:
                        self.ui.log(f"Skipped {name} (feature not available)", "WARNING")
                        progress.update(task, advance=1)
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
    
    def load_core(self):
        """PHASE 3: Load a saved model for inference."""
        self.ui.show_header("LOAD CORE", "Loading saved neural architecture...")
        
        # Get list of saved models
        models_dir = str(self.config.OUTPUT_DIR / 'models')
        saved_models = list_saved_models(models_dir)
        
        if not saved_models:
            self.ui.log("No saved models found!", "ERROR")
            self.ui.log(f"Models directory: {models_dir}", "INFO")
            self.ui.pause()
            return False
        
        # Show model selection
        selected_idx = self.ui.show_model_selection(saved_models)
        
        if selected_idx is None or selected_idx < 0 or selected_idx >= len(saved_models):
            self.ui.log("Model selection cancelled", "WARNING")
            self.ui.pause()
            return False
        
        # Load selected model
        selected_model = saved_models[selected_idx]
        self.ui.log(f"Loading model: {selected_model['filename']}", "SYSTEM")
        
        try:
            # Show loading animation
            self.ui.show_loading_animation("Loading neural core", duration=2)
            
            # Load model
            model_data = load_model(
                filepath=selected_model['filepath'],
                model_class=DynamicMLP
            )
            
            if model_data:
                self.loaded_model, self.loaded_genome, self.loaded_history, self.loaded_metrics = model_data
                
                # Display info
                self.ui.log("Model loaded successfully!", "SUCCESS")
                self.ui.log(f"  Fitness: {selected_model.get('fitness', 'N/A')}", "INFO")
                self.ui.log(f"  Accuracy: {selected_model.get('accuracy', 'N/A')}", "INFO")
                self.ui.log(f"  Timestamp: {selected_model.get('timestamp', 'N/A')}", "INFO")
                
                # Show architecture
                if self.loaded_genome:
                    arch = ' → '.join([str(n) for n in self.loaded_genome.hidden_layers])
                    self.ui.log(f"  Architecture: {arch}", "INFO")
                
                self.ui.pause()
                return True
            else:
                self.ui.log("Failed to load model!", "ERROR")
                self.ui.pause()
                return False
                
        except Exception as e:
            self.ui.log(f"Error loading model: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            self.ui.pause()
            return False
    
    def run_inference(self):
        """PHASE 3: Run inference mode on random samples."""
        if self.loaded_model is None:
            self.ui.log("No model loaded! Load a model first (option 2).", "ERROR")
            self.ui.pause()
            return
        
        self.ui.show_header("INFERENCE MODE", "Testing model on random samples...")
        
        # Load data if not already loaded
        if self.data_handler is None:
            self.ui.log("Loading dataset for inference...", "SYSTEM")
            data_path = self.config.DATASET_PATH if self.config.DATASET_PATH.exists() else None
            self.data_handler = DataHandler(
                data_path=str(data_path) if data_path else None,
                task=self.config.TASK
            )
            if not self.data_handler.load_data():
                self.ui.log("Failed to load dataset!", "ERROR")
                self.ui.pause()
                return
        
        # Show scanning animation
        self.ui.show_inference_scanning()
        
        # Get test data
        data_splits = self.data_handler.get_data_splits()
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        # Select 5 random samples
        n_samples = min(5, len(X_test))
        random_indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        # Prepare results
        inference_results = []
        
        self.ui.log("Running predictions...", "SYSTEM")
        
        for idx in random_indices:
            sample_x = X_test[idx:idx+1]
            true_label = y_test[idx]
            
            # Get prediction and confidence
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                prediction = self.loaded_model.predict(sample_x)[0]
                
                # Get probabilities for confidence
                self.loaded_model.eval()
                import torch
                with torch.no_grad():
                    device = next(self.loaded_model.parameters()).device
                    sample_tensor = torch.FloatTensor(sample_x).to(device)
                    output = self.loaded_model(sample_tensor)
                    
                    if self.config.TASK == 'classification':
                        probs = torch.nn.functional.softmax(output, dim=1)
                        confidence = probs[0][prediction].item() * 100
                    else:
                        confidence = 100.0  # For regression, use 100%
            
            # Determine status
            is_match = (prediction == true_label)
            
            inference_results.append({
                'sample_num': len(inference_results) + 1,
                'features': sample_x[0].tolist(),
                'true_label': int(true_label),
                'prediction': int(prediction),
                'confidence': confidence,
                'match': is_match
            })
        
        # Display results
        self.ui.show_inference_results(inference_results)
        
        # Calculate accuracy
        accuracy = sum(1 for r in inference_results if r['match']) / len(inference_results) * 100
        self.ui.log(f"\nSample Accuracy: {accuracy:.2f}% ({sum(1 for r in inference_results if r['match'])}/{len(inference_results)} correct)", 
                   "SUCCESS" if accuracy >= 60 else "WARNING")
        
        self.ui.pause()
    
    def view_models(self):
        """PHASE 3: Display all saved models."""
        self.ui.show_header("SAVED MODELS", "Available neural cores...")
        
        models_dir = str(self.config.OUTPUT_DIR / 'models')
        saved_models = list_saved_models(models_dir)
        
        if not saved_models:
            self.ui.log("No saved models found!", "WARNING")
            self.ui.log(f"Models directory: {models_dir}", "INFO")
        else:
            # Create display table
            from rich.table import Table
            from rich.box import DOUBLE
            
            table = Table(
                title="[bold cyan]SAVED NEURAL CORES[/]",
                box=DOUBLE,
                border_style="bright_blue"
            )
            
            table.add_column("#", style="yellow", justify="center")
            table.add_column("Filename", style="cyan")
            table.add_column("Timestamp", style="magenta")
            table.add_column("Fitness", style="green", justify="right")
            table.add_column("Accuracy", style="bright_magenta", justify="right")
            
            for idx, model in enumerate(saved_models, 1):
                table.add_row(
                    str(idx),
                    model['filename'][:50],
                    model['timestamp'][:19] if len(model['timestamp']) > 19 else model['timestamp'],
                    f"{model.get('fitness', 0):.4f}",
                    f"{model.get('accuracy', 0):.4f}"
                )
            
            self.ui.console.print(table)
            self.ui.log(f"\nTotal models: {len(saved_models)}", "INFO")
        
        self.ui.pause()


def main():
    """Main entry point with PHASE 3 menu-driven interface."""
    # Create config
    config = Config()
    
    # Save default config
    config.save_to_yaml()
    
    # Create system
    system = VinoGenCyberCore(config)
    
    # PHASE 3: Menu-driven loop
    while True:
        try:
            # Show main menu
            choice = system.ui.show_main_menu()
            
            if choice == '1':
                # NEW RUN - Execute full pipeline
                system.run()
                
            elif choice == '2':
                # LOAD CORE - Load saved model
                system.load_core()
                
            elif choice == '3':
                # INFERENCE - Run predictions
                system.run_inference()
                
            elif choice == '4':
                # VIEW MODELS - List saved models
                system.view_models()
                
            elif choice == '5':
                # EXIT - Shutdown
                system.ui.show_completion_banner()
                system.ui.log("System shutdown complete", "SUCCESS")
                break
            
            else:
                system.ui.log("Invalid choice", "ERROR")
                time.sleep(1)
                
        except KeyboardInterrupt:
            system.ui.log("\nInterrupted by user", "WARNING")
            system.ui.show_completion_banner()
            break
        except Exception as e:
            system.ui.log(f"Unexpected error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            system.ui.pause()


if __name__ == "__main__":
    main()
