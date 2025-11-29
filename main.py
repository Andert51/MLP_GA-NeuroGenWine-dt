"""
VinoGen-CyberCore: Main Orchestrator
The ultimate neuroevolution system for wine quality prediction.

This system combines:
- Genetic Algorithms for neural architecture search
- PyTorch-based MLP with dynamic topology
- Cyberpunk-themed CLI interface
- Comprehensive visualizations

Author: Cerebros.cpp team
Version: 1.9.0
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
from src.visualization import Visualizer, DatasetAnalyzer, AdvancedVisualizer, ModelExplainer, AnimationGenerator
from src.ui import CyberpunkUI
from src.utils import (
    Config, MATH_EQUATIONS, set_random_seeds, 
    EarlyStopping, save_json, get_device,
    save_model, load_model, list_saved_models,
    create_markdown_report, ensure_directories,
    MetricsTracker, PerformanceTimer
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
        
        # New advanced visualizers
        self.dataset_analyzer = DatasetAnalyzer(output_dir=self.config.OUTPUT_DIR / "analysis")
        self.advanced_viz = AdvancedVisualizer(output_dir=self.config.OUTPUT_DIR / "advanced")
        self.explainer = ModelExplainer(output_dir=self.config.OUTPUT_DIR / "explanations")
        self.animator = AnimationGenerator(output_dir=self.config.OUTPUT_DIR / "animations")
        
        self.device = get_device()
        
        # Performance tracking
        self.metrics_tracker = MetricsTracker(task=self.config.TASK)
        self.perf_timer = PerformanceTimer()
        
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
        
        # NEW: Training curves animation GIF
        if 'training_history' in self.results:
            viz_tasks.append(("Training Animation GIF", lambda: self.animator.create_training_animation(
                self.results['training_history'],
                filename="training_evolution.gif",
                fps=10
            )))
        
        # NEW: Genetic evolution animation
        if 'evolution_history' in self.results and 'population_data' in self.results['evolution_history']:
            # Use actual population data
            gen_data = []
            for pop_data in self.results['evolution_history']['population_data']:
                gen_data.append({
                    'generation': pop_data['generation'] + 1,
                    'fitness_scores': pop_data['fitness_scores'],
                    'best_fitness': max(pop_data['fitness_scores']),
                    'avg_fitness': np.mean(pop_data['fitness_scores']),
                    'architectures': pop_data['architectures']
                })
            
            if gen_data:
                viz_tasks.append(("Genetic Evolution GIF", lambda: self.animator.create_genetic_evolution_animation(
                    gen_data,
                    filename="genetic_evolution.gif",
                    fps=3
                )))
        elif 'evolution_history' in self.results:
            # Fallback: use basic history data
            gen_data = []
            history = self.results['evolution_history']
            for i in range(len(history['best_fitness'])):
                gen_data.append({
                    'generation': i + 1,
                    'fitness_scores': [history['best_fitness'][i]] * 20,  # Placeholder
                    'best_fitness': history['best_fitness'][i],
                    'avg_fitness': history['avg_fitness'][i],
                    'architectures': []
                })
            
            if gen_data:
                viz_tasks.append(("Genetic Evolution GIF", lambda: self.animator.create_genetic_evolution_animation(
                    gen_data,
                    filename="genetic_evolution.gif",
                    fps=3
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
    
    def explain_model(self):
        """
        Generate educational visualizations explaining what the model does.
        
        Shows:
        - What task the model performs (classification vs regression)
        - How inputs become outputs
        - Example predictions with detailed breakdown
        - Comparison between classification and regression
        - Instructions for using the model
        """
        self.ui.show_header(
            "MODEL EXPLANATION",
            "Visual explanation of what this model does"
        )
        
        try:
            # Load data if not already loaded
            if self.data_handler is None:
                self.ui.log("Loading dataset...", "INFO")
                data_path = self.config.DATASET_PATH if self.config.DATASET_PATH.exists() else None
                self.data_handler = DataHandler(
                    data_path=str(data_path) if data_path else None,
                    task=self.config.TASK
                )
                if not self.data_handler.load_data():
                    self.ui.log("Failed to load dataset", "ERROR")
                    self.ui.pause()
                    return
            
            self.ui.log("\nGenerating educational visualizations...", "SYSTEM")
            paths = []
            
            # 1. Main explanation diagram
            self.ui.log("\n[1/2] Creating model task explanation...", "INFO")
            path1 = self.explainer.explain_model_task(
                task=self.config.TASK,
                n_classes=self.data_handler.n_classes if self.config.TASK == "classification" else 1,
                feature_names=[
                    'Acidez Fija', 'Acidez Volátil', 'Ácido Cítrico',
                    'Azúcar Residual', 'Cloruros', 'SO₂ Libre',
                    'SO₂ Total', 'Densidad', 'pH', 'Sulfatos', 'Alcohol'
                ]
            )
            paths.append(('Model Task Explanation', path1))
            self.ui.log(f"  ✓ Saved: {path1}", "SUCCESS")
            
            # 2. Example prediction (if model exists)
            if self.loaded_model is not None or self.best_model is not None:
                self.ui.log("\n[2/2] Creating example prediction breakdown...", "INFO")
                
                model = self.loaded_model if self.loaded_model is not None else self.best_model
                
                # Get a random test sample
                data_splits = self.data_handler.get_data_splits()
                idx = np.random.randint(0, len(data_splits['X_test']))
                X_sample = data_splits['X_test'][idx]
                y_true = data_splits['y_test'][idx]
                
                class_names = None
                if self.config.TASK == "classification":
                    if self.data_handler.n_classes == 3:
                        class_names = ['BAJA', 'MEDIA', 'ALTA']
                    else:
                        class_names = [f'Clase {i}' for i in range(self.data_handler.n_classes)]
                
                path2 = self.explainer.explain_prediction_example(
                    model=model,
                    X_sample=X_sample,
                    y_true=y_true,
                    feature_names=[
                        'Acidez Fija', 'Acidez Volátil', 'Ácido Cítrico',
                        'Azúcar Residual', 'Cloruros', 'SO₂ Libre',
                        'SO₂ Total', 'Densidad', 'pH', 'Sulfatos', 'Alcohol'
                    ],
                    task=self.config.TASK,
                    class_names=class_names
                )
                paths.append(('Prediction Example', path2))
                self.ui.log(f"  ✓ Saved: {path2}", "SUCCESS")
            else:
                self.ui.log("\n[2/2] Skipping prediction example (no model loaded)", "WARNING")
            
            # Summary
            self.ui.log("\n" + "="*60, "SYSTEM")
            self.ui.log("MODEL EXPLANATION COMPLETE!", "SUCCESS")
            self.ui.log("="*60, "SYSTEM")
            
            self.ui.log(f"\nTotal visualizations created: {len(paths)}", "INFO")
            
            # Display paths table
            from rich.table import Table
            table = Table(title="\n📊 Generated Explanations")
            table.add_column("#", style="cyan", justify="center")
            table.add_column("Type", style="magenta")
            table.add_column("Path", style="green")
            
            for i, (viz_type, path) in enumerate(paths, 1):
                table.add_row(str(i), viz_type, str(path))
            
            self.ui.console.print(table)
            
            self.ui.log(f"\nAll explanations saved in: {self.config.OUTPUT_DIR / 'explanations'}/", "SUCCESS")
            
            # Explanation text
            explanation = f"""

📖 QUÉ MUESTRA CADA VISUALIZACIÓN:

1. MODEL TASK EXPLANATION:
   • Qué hace el modelo (clasificación de {self.data_handler.n_classes} clases o regresión)
   • Cómo procesa las 11 características químicas
   • Qué significa cada salida
   • Comparación clasificación vs regresión
   • Instrucciones de uso

2. PREDICTION EXAMPLE (si hay modelo):
   • Desglose detallado de una predicción
   • Valores de entrada mostrados
   • Salida del modelo explicada
   • Importancia de cada característica
   • Resultado final con interpretación

💡 TU MODELO ACTUAL:
   Tarea: {self.config.TASK.upper()}
   {'Clasifica vinos en ' + str(self.data_handler.n_classes) + ' categorías de calidad' if self.config.TASK == 'classification' else 'Predice calidad exacta (0-10)'}

🔧 PARA CAMBIAR A {'REGRESIÓN' if self.config.TASK == 'classification' else 'CLASIFICACIÓN'}:
   Edita: src/utils/config.py
   Cambia: TASK = "{'regression' if self.config.TASK == 'classification' else 'classification'}"
"""
            
            self.ui.log(explanation, "INFO")
            
        except Exception as e:
            self.ui.log(f"Error generating explanations: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        
        self.ui.pause()
    
    def interactive_test(self):
        """Test model with user-provided wine characteristics."""
        self.ui.clear_screen()
        self.ui.show_header("INTERACTIVE WINE TEST", "Test model with custom wine sample")
        
        # Check if model is loaded
        model = self.loaded_model if self.loaded_model else self.best_model
        
        if model is None:
            self.ui.log("No model available! Please run NEW RUN or LOAD CORE first.", "ERROR")
            self.ui.pause()
            return
        
        # Check if data handler exists (for feature names)
        if self.data_handler is None:
            self.ui.log("Loading dataset for feature information...", "SYSTEM")
            data_path = self.config.DATASET_PATH if self.config.DATASET_PATH.exists() else None
            self.data_handler = DataHandler(
                data_path=str(data_path) if data_path else None,
                task=self.config.TASK
            )
            if not self.data_handler.load_data():
                self.ui.log("Failed to load dataset!", "ERROR")
                self.ui.pause()
                return
        
        # Get features from user
        features_dict = self.ui.get_wine_features_interactive()
        
        if features_dict is None:
            self.ui.log("Test cancelled", "WARNING")
            return
        
        # Convert to numpy array in correct order
        feature_names = [
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "pH", "sulphates", "alcohol"
        ]
        features_array = np.array([[features_dict[name] for name in feature_names]])
        
        # Normalize features (using data handler's scaler)
        if hasattr(self.data_handler, 'scaler') and self.data_handler.scaler is not None:
            features_normalized = self.data_handler.scaler.transform(features_array)
        else:
            features_normalized = features_array
        
        # Make prediction
        try:
            import torch
            model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_normalized).to(self.device)
                output = model(features_tensor)
                
                if self.config.TASK == "classification":
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                    prediction = int(torch.argmax(output, dim=1).cpu().numpy()[0])
                    
                    self.ui.show_prediction_result(
                        features_dict, prediction, probabilities, 
                        task=self.config.TASK
                    )
                else:
                    prediction = float(output.cpu().numpy()[0][0])
                    
                    self.ui.show_prediction_result(
                        features_dict, prediction, 
                        task=self.config.TASK
                    )
            
            self.ui.log("\n✓ Prediction completed successfully!", "SUCCESS")
            
        except Exception as e:
            self.ui.log(f"Error making prediction: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        
        self.ui.pause()
    
    def toggle_mode(self):
        """Toggle between classification and regression modes."""
        current_mode = self.config.TASK
        
        # Show confirmation dialog
        if not self.ui.show_mode_toggle(current_mode):
            self.ui.log("Mode toggle cancelled", "WARNING")
            time.sleep(1)
            return
        
        # Toggle the mode
        new_mode = "regression" if current_mode == "classification" else "classification"
        self.config.TASK = new_mode
        
        # Update config file
        config_path = Path(__file__).parent / "src" / "utils" / "config.py"
        
        try:
            # Read current config
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find and replace TASK line
            for i, line in enumerate(lines):
                if line.strip().startswith('TASK ='):
                    lines[i] = f'    TASK = "{new_mode}"  # or "regression"\n'
                    break
            
            # Write back
            with open(config_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            self.ui.log(f"\n✓ Mode switched to {new_mode.upper()}!", "SUCCESS")
            self.ui.log(f"Config file updated: {config_path}", "INFO")
            self.ui.log("\nNote: You need to run NEW RUN to train a model in the new mode", "WARNING")
            
            # Reset any loaded models since they're for the wrong task
            if self.loaded_model or self.best_model:
                self.ui.log("Clearing loaded models (incompatible with new mode)", "WARNING")
                self.loaded_model = None
                self.best_model = None
            
            # Reset data handler to reload with new task
            self.data_handler = None
            self.metrics_tracker = MetricsTracker(task=new_mode)
            
        except Exception as e:
            self.ui.log(f"Error updating config: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        
        self.ui.pause()
    
    def deep_analysis(self):
        """PHASE 3: Perform deep analysis of dataset and model."""
        self.ui.show_header("DEEP ANALYSIS", "Advanced Dataset & Model Analysis")
        
        # Check if we have data
        if self.data_handler is None:
            self.ui.log("Loading dataset for analysis...", "SYSTEM")
            data_path = self.config.DATASET_PATH if self.config.DATASET_PATH.exists() else None
            self.data_handler = DataHandler(
                data_path=str(data_path) if data_path else None,
                task=self.config.TASK
            )
            if not self.data_handler.load_data():
                self.ui.log("Failed to load dataset!", "ERROR")
                self.ui.pause()
                return
        
        # Get data
        data_splits = self.data_handler.get_data_splits()
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        self.ui.log("Starting comprehensive analysis...", "SYSTEM")
        visualizations_created = []
        
        try:
            # 1. Dataset Overview
            self.ui.log("\n[1/8] Generating dataset overview...", "INFO")
            overview_path = self.dataset_analyzer.plot_dataset_overview(
                X_train, y_train,
                feature_names=self.data_handler.feature_names if hasattr(self.data_handler, 'feature_names') else None
            )
            visualizations_created.append(("Dataset Overview", overview_path))
            self.ui.log(f"  ✓ Saved: {overview_path}", "SUCCESS")
            
            # 2. Sample Visualization
            self.ui.log("[2/8] Visualizing dataset samples...", "INFO")
            samples_path = self.dataset_analyzer.plot_sample_visualization(
                X_train, y_train, n_samples=10,
                feature_names=self.data_handler.feature_names if hasattr(self.data_handler, 'feature_names') else None
            )
            visualizations_created.append(("Sample Visualization", samples_path))
            self.ui.log(f"  ✓ Saved: {samples_path}", "SUCCESS")
            
            # Check if we have a model
            model_to_analyze = self.loaded_model if self.loaded_model else self.best_model
            
            if model_to_analyze:
                self.ui.log("\nModel found! Generating model visualizations...", "SYSTEM")
                
                # 3. Decision Boundary 2D (PCA)
                self.ui.log("[3/8] Creating 2D decision boundary (PCA)...", "INFO")
                db_2d_pca = self.advanced_viz.plot_decision_boundary_2d(
                    model_to_analyze, X_test, y_test, reduction_method='pca'
                )
                visualizations_created.append(("Decision Boundary 2D (PCA)", db_2d_pca))
                self.ui.log(f"  ✓ Saved: {db_2d_pca}", "SUCCESS")
                
                # 4. Decision Boundary 2D (t-SNE)
                self.ui.log("[4/8] Creating 2D decision boundary (t-SNE)...", "INFO")
                db_2d_tsne = self.advanced_viz.plot_decision_boundary_2d(
                    model_to_analyze, X_test, y_test, reduction_method='tsne'
                )
                visualizations_created.append(("Decision Boundary 2D (t-SNE)", db_2d_tsne))
                self.ui.log(f"  ✓ Saved: {db_2d_tsne}", "SUCCESS")
                
                # 5. Decision Boundary 3D
                self.ui.log("[5/8] Creating 3D decision boundary...", "INFO")
                db_3d = self.advanced_viz.plot_decision_boundary_3d(
                    model_to_analyze, X_test, y_test
                )
                visualizations_created.append(("Decision Boundary 3D", db_3d))
                self.ui.log(f"  ✓ Saved: {db_3d}", "SUCCESS")
                
                # 6. Activation Heatmap 3D (first layer)
                self.ui.log("[6/8] Generating activation heatmap 3D...", "INFO")
                act_heatmap = self.advanced_viz.plot_activation_heatmap_3d(
                    model_to_analyze, X_test[:100], layer_idx=0
                )
                visualizations_created.append(("Activation Heatmap 3D", act_heatmap))
                self.ui.log(f"  ✓ Saved: {act_heatmap}", "SUCCESS")
                
                # 7. Weight Heatmap 2D (first layer)
                self.ui.log("[7/8] Creating weight heatmap 2D...", "INFO")
                weight_heatmap = self.advanced_viz.plot_weight_heatmap_2d(
                    model_to_analyze, layer_idx=0
                )
                visualizations_created.append(("Weight Heatmap 2D", weight_heatmap))
                self.ui.log(f"  ✓ Saved: {weight_heatmap}", "SUCCESS")
                
                # 8. Prediction Confidence Analysis
                self.ui.log("[8/8] Analyzing prediction confidence...", "INFO")
                model_to_analyze.eval()
                import torch
                with torch.no_grad():
                    y_pred = model_to_analyze.predict(X_test)
                    
                    # Get confidences
                    X_test_tensor = torch.FloatTensor(X_test).to(
                        next(model_to_analyze.parameters()).device
                    )
                    outputs = model_to_analyze(X_test_tensor)
                    
                    if self.config.TASK == 'classification':
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        confidences = probs.max(dim=1)[0].cpu().numpy()
                    else:
                        confidences = np.ones(len(y_pred))  # For regression
                
                conf_dist = self.advanced_viz.plot_prediction_confidence_distribution(
                    y_test, y_pred, confidences
                )
                visualizations_created.append(("Confidence Analysis", conf_dist))
                self.ui.log(f"  ✓ Saved: {conf_dist}", "SUCCESS")
            else:
                self.ui.log("\nNo model available. Skipping model visualizations.", "WARNING")
                self.ui.log("Run option [1] NEW RUN or [2] LOAD CORE first.", "INFO")
            
            # Summary
            self.ui.log("\n" + "="*60, "SUCCESS")
            self.ui.log("DEEP ANALYSIS COMPLETE!", "SUCCESS")
            self.ui.log("="*60, "SUCCESS")
            self.ui.log(f"\nTotal visualizations created: {len(visualizations_created)}", "INFO")
            
            # Display list
            from rich.table import Table
            from rich.box import ROUNDED
            
            table = Table(
                title="[bold cyan]Generated Visualizations[/]",
                box=ROUNDED,
                border_style="cyan"
            )
            table.add_column("#", style="yellow", justify="center")
            table.add_column("Type", style="cyan")
            table.add_column("Path", style="green")
            
            for idx, (viz_type, viz_path) in enumerate(visualizations_created, 1):
                # Shorten path for display
                short_path = str(viz_path).replace(str(self.config.OUTPUT_DIR), "output")
                table.add_row(str(idx), viz_type, short_path)
            
            self.ui.console.print("\n")
            self.ui.console.print(table)
            
            self.ui.log(f"\nAll visualizations saved in: {self.config.OUTPUT_DIR}", "INFO")
            
        except Exception as e:
            self.ui.log(f"Error during analysis: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        
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
                # DEEP ANALYSIS - Advanced analysis
                system.deep_analysis()
            
            elif choice == '6':
                # EXPLAIN MODEL - Educational visualizations
                system.explain_model()
            
            elif choice == '7':
                # INTERACTIVE TEST - Test with custom sample
                system.interactive_test()
            
            elif choice == '8':
                # TOGGLE MODE - Switch classification/regression
                system.toggle_mode()
                
            elif choice == '9':
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
