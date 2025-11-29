"""
NeuroGen WineLab: Animation Generator
Creates animated GIFs for training visualization and decision boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from pathlib import Path
import torch
from typing import List, Tuple, Optional
import seaborn as sns


class AnimationGenerator:
    """
    Generate animated visualizations for model training and evolution.
    
    Features:
    - Training curves animation (loss/accuracy over epochs)
    - Decision boundary evolution during training
    - Genetic algorithm evolution visualization
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize animation generator.
        
        Args:
            output_dir: Directory to save animations
        """
        self.output_dir = Path(output_dir) if output_dir else Path("output/animations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cyberpunk color scheme
        self.colors = {
            'bg': '#1a1a2e',
            'primary': '#00ff9f',
            'secondary': '#00d9ff',
            'accent': '#ff006e',
            'warning': '#ffaa00',
            'grid': '#2a2a3e'
        }
        
        plt.style.use('dark_background')
    
    def create_training_animation(self, 
                                 history: dict,
                                 filename: str = "training_animation.gif",
                                 fps: int = 10) -> Path:
        """
        Create animated GIF showing training progress over epochs.
        
        Args:
            history: Training history with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            filename: Output filename
            fps: Frames per second
            
        Returns:
            Path to saved GIF
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor=self.colors['bg'])
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Initialize empty lines
        train_loss_line, = ax1.plot([], [], color=self.colors['primary'], 
                                     linewidth=2, label='Train Loss')
        val_loss_line, = ax1.plot([], [], color=self.colors['accent'], 
                                   linewidth=2, label='Val Loss', linestyle='--')
        
        train_acc_line, = ax2.plot([], [], color=self.colors['secondary'], 
                                    linewidth=2, label='Train Accuracy')
        val_acc_line, = ax2.plot([], [], color=self.colors['warning'], 
                                  linewidth=2, label='Val Accuracy', linestyle='--')
        
        # Configure axes
        ax1.set_facecolor(self.colors['bg'])
        ax2.set_facecolor(self.colors['bg'])
        
        ax1.set_xlim(0, len(epochs) + 1)
        ax1.set_ylim(0, max(history['train_loss']) * 1.1)
        ax1.set_xlabel('Epoch', color='white', fontsize=12)
        ax1.set_ylabel('Loss', color='white', fontsize=12)
        ax1.set_title('Training Loss Evolution', color=self.colors['primary'], 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='upper right', framealpha=0.8)
        ax1.grid(True, alpha=0.2, color=self.colors['grid'])
        
        ax2.set_xlim(0, len(epochs) + 1)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel('Epoch', color='white', fontsize=12)
        ax2.set_ylabel('Accuracy', color='white', fontsize=12)
        ax2.set_title('Training Accuracy Evolution', color=self.colors['secondary'], 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='lower right', framealpha=0.8)
        ax2.grid(True, alpha=0.2, color=self.colors['grid'])
        
        plt.tight_layout()
        
        def animate(frame):
            """Update function for animation."""
            # Update data up to current frame
            x_data = list(epochs[:frame+1])
            
            train_loss_line.set_data(x_data, history['train_loss'][:frame+1])
            val_loss_line.set_data(x_data, history['val_loss'][:frame+1])
            
            train_acc_line.set_data(x_data, history['train_acc'][:frame+1])
            val_acc_line.set_data(x_data, history['val_acc'][:frame+1])
            
            # Add current epoch marker
            ax1.set_title(f'Training Loss Evolution - Epoch {frame+1}/{len(epochs)}', 
                         color=self.colors['primary'], fontsize=14, fontweight='bold', pad=15)
            
            return train_loss_line, val_loss_line, train_acc_line, val_acc_line
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(epochs), 
                                      interval=1000//fps, blit=True, repeat=True)
        
        # Save as GIF
        output_path = self.output_dir / filename
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close(fig)
        
        return output_path
    
    def create_decision_boundary_animation(self,
                                          model_checkpoints: List[Tuple[any, int]],
                                          X: np.ndarray,
                                          y: np.ndarray,
                                          filename: str = "decision_boundary_animation.gif",
                                          fps: int = 2,
                                          device: str = 'cpu') -> Path:
        """
        Create animated GIF showing decision boundary evolution during training.
        
        Args:
            model_checkpoints: List of (model_state, epoch) tuples
            X: Feature data
            y: Labels
            filename: Output filename
            fps: Frames per second
            device: Device for model inference
            
        Returns:
            Path to saved GIF
        """
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.colors['bg'])
        ax.set_facecolor(self.colors['bg'])
        
        # Set up mesh for decision boundary
        h = 0.02  # step size
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        def animate(frame):
            """Update function for animation."""
            ax.clear()
            ax.set_facecolor(self.colors['bg'])
            
            model_state, epoch = model_checkpoints[frame]
            
            # Reconstruct model and load state
            # Note: This assumes model architecture is available
            # You may need to pass model architecture info
            
            # For now, create contour plot placeholder
            # In production, you'd predict on mesh and create contourf
            
            # Plot data points
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, 
                               cmap='viridis', alpha=0.6, edgecolors='white',
                               linewidth=0.5, s=50)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('PCA Component 1', color='white', fontsize=12)
            ax.set_ylabel('PCA Component 2', color='white', fontsize=12)
            ax.set_title(f'Decision Boundary Evolution - Epoch {epoch}', 
                        color=self.colors['primary'], fontsize=14, 
                        fontweight='bold', pad=15)
            ax.grid(True, alpha=0.2, color=self.colors['grid'])
            
            return scatter,
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(model_checkpoints),
                                      interval=1000//fps, blit=False, repeat=True)
        
        # Save as GIF
        output_path = self.output_dir / filename
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close(fig)
        
        return output_path
    
    def create_genetic_evolution_animation(self,
                                          generation_data: List[dict],
                                          filename: str = "genetic_evolution.gif",
                                          fps: int = 3) -> Path:
        """
        Create animated visualization of genetic algorithm evolution.
        
        Args:
            generation_data: List of dicts with 'generation', 'fitness_scores', 
                           'best_fitness', 'avg_fitness', 'architectures'
            filename: Output filename
            fps: Frames per second
            
        Returns:
            Path to saved GIF
        """
        fig = plt.figure(figsize=(14, 10), facecolor=self.colors['bg'])
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])  # Fitness evolution
        ax2 = fig.add_subplot(gs[1, 0])  # Fitness distribution
        ax3 = fig.add_subplot(gs[1, 1])  # Architecture diversity
        ax4 = fig.add_subplot(gs[2, :])  # Population fitness
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(self.colors['bg'])
        
        def animate(frame):
            """Update function for animation."""
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
                ax.set_facecolor(self.colors['bg'])
            
            current_gen = generation_data[frame]
            gen_num = current_gen['generation']
            
            # 1. Fitness evolution over generations
            gens = [g['generation'] for g in generation_data[:frame+1]]
            best_fits = [g['best_fitness'] for g in generation_data[:frame+1]]
            avg_fits = [g['avg_fitness'] for g in generation_data[:frame+1]]
            
            ax1.plot(gens, best_fits, color=self.colors['primary'], 
                    linewidth=2, marker='o', label='Best Fitness')
            ax1.plot(gens, avg_fits, color=self.colors['secondary'], 
                    linewidth=2, marker='s', label='Avg Fitness', alpha=0.7)
            ax1.set_xlabel('Generation', color='white', fontsize=11)
            ax1.set_ylabel('Fitness', color='white', fontsize=11)
            ax1.set_title(f'Fitness Evolution - Generation {gen_num}', 
                         color=self.colors['primary'], fontsize=13, fontweight='bold')
            ax1.legend(loc='lower right', framealpha=0.8)
            ax1.grid(True, alpha=0.2, color=self.colors['grid'])
            
            # 2. Current generation fitness distribution
            fitness_scores = current_gen['fitness_scores']
            ax2.hist(fitness_scores, bins=15, color=self.colors['accent'], 
                    alpha=0.7, edgecolor='white', linewidth=0.5)
            ax2.axvline(current_gen['best_fitness'], color=self.colors['primary'], 
                       linestyle='--', linewidth=2, label='Best')
            ax2.axvline(current_gen['avg_fitness'], color=self.colors['secondary'], 
                       linestyle='--', linewidth=2, label='Average')
            ax2.set_xlabel('Fitness', color='white', fontsize=11)
            ax2.set_ylabel('Count', color='white', fontsize=11)
            ax2.set_title('Fitness Distribution', color=self.colors['accent'], 
                         fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right', framealpha=0.8, fontsize=9)
            ax2.grid(True, alpha=0.2, color=self.colors['grid'], axis='y')
            
            # 3. Architecture diversity (layers and neurons)
            if 'architectures' in current_gen:
                archs = current_gen['architectures']
                num_layers = [len(a['hidden_layers']) for a in archs]
                total_neurons = [sum(a['hidden_layers']) for a in archs]
                
                ax3.scatter(num_layers, total_neurons, c=fitness_scores, 
                          cmap='plasma', s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
                ax3.set_xlabel('Number of Layers', color='white', fontsize=11)
                ax3.set_ylabel('Total Neurons', color='white', fontsize=11)
                ax3.set_title('Architecture Diversity', color=self.colors['warning'], 
                            fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.2, color=self.colors['grid'])
            
            # 4. Population fitness bar chart
            sorted_fitness = sorted(enumerate(fitness_scores), key=lambda x: x[1], reverse=True)
            indices, values = zip(*sorted_fitness)
            colors_bar = [self.colors['primary'] if i == 0 else 
                         self.colors['secondary'] if i < 3 else 
                         self.colors['grid'] for i in range(len(values))]
            
            ax4.bar(range(len(values)), values, color=colors_bar, alpha=0.8, 
                   edgecolor='white', linewidth=0.5)
            ax4.set_xlabel('Individual (sorted by fitness)', color='white', fontsize=11)
            ax4.set_ylabel('Fitness', color='white', fontsize=11)
            ax4.set_title('Population Ranking', color=self.colors['secondary'], 
                         fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.2, color=self.colors['grid'], axis='y')
            
            # Add statistics text
            stats_text = f"Best: {current_gen['best_fitness']:.4f} | "
            stats_text += f"Avg: {current_gen['avg_fitness']:.4f} | "
            stats_text += f"Pop: {len(fitness_scores)}"
            fig.text(0.5, 0.02, stats_text, ha='center', color=self.colors['primary'],
                    fontsize=11, fontweight='bold')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(generation_data),
                                      interval=1000//fps, blit=False, repeat=True)
        
        # Save as GIF
        output_path = self.output_dir / filename
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close(fig)
        
        return output_path
    
    def create_simple_training_gif(self,
                                  train_losses: list,
                                  val_losses: list,
                                  train_accs: list,
                                  val_accs: list,
                                  filename: str = "training_curves.gif",
                                  fps: int = 10) -> Path:
        """
        Simplified training animation creator.
        
        Args:
            train_losses: Training loss per epoch
            val_losses: Validation loss per epoch
            train_accs: Training accuracy per epoch
            val_accs: Validation accuracy per epoch
            filename: Output filename
            fps: Frames per second
            
        Returns:
            Path to saved GIF
        """
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        }
        
        return self.create_training_animation(history, filename, fps)
