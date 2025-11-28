"""
VinoGen-CyberCore: Visualization Engine
Creates stunning visuals: topology graphs, learning curves, 3D landscapes, and neural animations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import imageio
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """
    Advanced visualization system for neural network analysis.
    
    Capabilities:
    - Network topology graphs
    - Neuron activation flow animations
    - Learning curve plots
    - Confusion matrices
    - 3D loss landscapes
    - Decision boundaries
    - Probability heatmaps
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the Visualization Engine.
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Cyberpunk color scheme
        self.colors = {
            'neon_green': '#39FF14',
            'electric_blue': '#00FFFF',
            'deep_purple': '#9D00FF',
            'hot_pink': '#FF10F0',
            'cyber_yellow': '#FFFF00',
            'dark_bg': '#0A0E27',
            'grid': '#1A1F3A'
        }
        
        # Set matplotlib style
        plt.style.use('dark_background')
        sns.set_palette([self.colors['neon_green'], self.colors['electric_blue'], 
                        self.colors['deep_purple'], self.colors['hot_pink']])
    
    def plot_network_topology(self, 
                              genome: Dict, 
                              input_dim: int, 
                              output_dim: int,
                              filename: str = "network_topology.png"):
        """
        Create a visual graph of the network architecture.
        
        Args:
            genome: Network architecture dictionary
            input_dim: Number of input features
            output_dim: Number of output neurons
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(16, 10), facecolor=self.colors['dark_bg'])
        ax.set_facecolor(self.colors['dark_bg'])
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Build layer structure
        layers = [input_dim] + genome['hidden_layers'] + [output_dim]
        
        # Calculate positions for aesthetic layout
        pos = {}
        node_colors = []
        node_sizes = []
        labels = {}
        
        node_id = 0
        max_neurons = max(layers)
        
        for layer_idx, layer_size in enumerate(layers):
            # Vertical spacing for this layer
            x = layer_idx / (len(layers) - 1)
            y_spacing = 1.0 / (layer_size + 1)
            
            # Center vertically
            y_offset = (max_neurons - layer_size) / (2 * max_neurons)
            
            for neuron_idx in range(layer_size):
                y = y_offset + (neuron_idx + 1) * y_spacing / max_neurons * layer_size
                pos[node_id] = (x, y)
                
                # Color by layer type
                if layer_idx == 0:
                    node_colors.append(self.colors['electric_blue'])
                    node_sizes.append(500)
                    labels[node_id] = f"I{neuron_idx}"
                elif layer_idx == len(layers) - 1:
                    node_colors.append(self.colors['hot_pink'])
                    node_sizes.append(600)
                    labels[node_id] = f"O{neuron_idx}"
                else:
                    node_colors.append(self.colors['neon_green'])
                    node_sizes.append(400)
                    labels[node_id] = ""
                
                G.add_node(node_id)
                node_id += 1
        
        # Add edges between consecutive layers
        node_id = 0
        for layer_idx in range(len(layers) - 1):
            layer_start = node_id
            layer_end = node_id + layers[layer_idx]
            next_layer_start = layer_end
            next_layer_end = layer_end + layers[layer_idx + 1]
            
            for i in range(layer_start, layer_end):
                for j in range(next_layer_start, next_layer_end):
                    G.add_edge(i, j)
            
            node_id += layers[layer_idx]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=self.colors['grid'], 
                              width=0.5, alpha=0.3, arrows=False, ax=ax)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                               font_color='white', ax=ax)
        
        # Add layer labels
        for i, (layer_size, layer_type) in enumerate(zip(layers, 
                ['INPUT'] + [f'HIDDEN {j+1}' for j in range(len(genome['hidden_layers']))] + ['OUTPUT'])):
            x = i / (len(layers) - 1)
            ax.text(x, -0.05, layer_type, ha='center', va='top', 
                   color=self.colors['cyber_yellow'], fontsize=12, fontweight='bold')
            ax.text(x, -0.10, f'{layer_size} neurons', ha='center', va='top',
                   color='white', fontsize=10)
            
            # Add activation function for hidden layers
            if 0 < i < len(layers) - 1:
                act = genome['activation_functions'][i-1]
                ax.text(x, 1.05, act.upper(), ha='center', va='bottom',
                       color=self.colors['deep_purple'], fontsize=10, 
                       style='italic')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, 1.1)
        ax.axis('off')
        
        # Title
        plt.title('NEURAL NETWORK TOPOLOGY', 
                 fontsize=24, color=self.colors['neon_green'], 
                 fontweight='bold', pad=20)
        
        # Save
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor=self.colors['dark_bg'])
        plt.close()
        
        return str(filepath)
    
    def create_activation_flow_animation(self,
                                        model,
                                        sample_input: np.ndarray,
                                        genome: Dict,
                                        filename: str = "activation_flow.gif",
                                        fps: int = 10):
        """
        Create animated GIF showing data flowing through the network.
        
        Args:
            model: Trained neural network
            sample_input: Single input sample
            genome: Network architecture
            filename: Output filename
            fps: Frames per second
        """
        # Get activations
        activations = model.get_activations(sample_input)
        
        # Create frames
        frames = []
        layers = [len(sample_input)] + genome['hidden_layers'] + [model.output_dim]
        
        for step in range(len(activations)):
            fig, axes = plt.subplots(1, len(activations), 
                                    figsize=(20, 4), 
                                    facecolor=self.colors['dark_bg'])
            
            if len(activations) == 1:
                axes = [axes]
            
            fig.suptitle(f'NEURON ACTIVATION PROPAGATION - Step {step + 1}/{len(activations)}',
                        fontsize=16, color=self.colors['neon_green'], fontweight='bold')
            
            for i, (ax, activation) in enumerate(zip(axes, activations)):
                ax.set_facecolor(self.colors['dark_bg'])
                
                # Normalize activations for visualization
                act = activation.flatten()
                act_norm = (act - act.min()) / (act.max() - act.min() + 1e-8)
                
                # Create heatmap
                if i <= step:
                    colors_map = plt.cm.plasma(act_norm)
                    ax.barh(range(len(act)), act_norm, color=colors_map, alpha=0.8)
                else:
                    ax.barh(range(len(act)), np.zeros(len(act)), 
                           color='gray', alpha=0.2)
                
                # Labels
                if i == 0:
                    ax.set_title('INPUT', color=self.colors['electric_blue'], fontweight='bold')
                elif i == len(activations) - 1:
                    ax.set_title('OUTPUT', color=self.colors['hot_pink'], fontweight='bold')
                else:
                    ax.set_title(f'HIDDEN {i}', color=self.colors['neon_green'], fontweight='bold')
                
                ax.set_xlim(0, 1)
                ax.set_ylabel('Neuron', color='white')
                ax.set_xlabel('Activation', color='white')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color(self.colors['grid'])
                ax.spines['left'].set_color(self.colors['grid'])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Convert to image (using buffer_rgba instead of deprecated tostring_rgb)
            fig.canvas.draw()
            # Use buffer_rgba() which is the modern API
            buf = fig.canvas.buffer_rgba()
            image = np.asarray(buf)[:, :, :3]  # Remove alpha channel
            frames.append(image)
            plt.close()
        
        # Add pause frames at the end
        for _ in range(fps):
            frames.append(frames[-1])
        
        # Save as GIF
        filepath = self.output_dir / filename
        imageio.mimsave(filepath, frames, fps=fps)
        
        return str(filepath)
    
    def plot_learning_curves(self,
                            history: Dict,
                            filename: str = "learning_curves.png"):
        """
        Plot training and validation metrics over epochs.
        
        Args:
            history: Training history dictionary
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), 
                                facecolor=self.colors['dark_bg'])
        
        for ax in axes:
            ax.set_facecolor(self.colors['dark_bg'])
        
        # Loss curves
        if 'train_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'], 
                        color=self.colors['neon_green'], linewidth=2, 
                        label='Training Loss', marker='o', markersize=4)
            
            if 'val_loss' in history:
                axes[0].plot(epochs, history['val_loss'], 
                           color=self.colors['electric_blue'], linewidth=2,
                           label='Validation Loss', marker='s', markersize=4)
            
            axes[0].set_xlabel('Epoch', fontsize=12, color='white')
            axes[0].set_ylabel('Loss', fontsize=12, color='white')
            axes[0].set_title('LOSS CURVES', fontsize=14, 
                            color=self.colors['cyber_yellow'], fontweight='bold')
            axes[0].legend(facecolor=self.colors['grid'], edgecolor='white')
            axes[0].grid(True, alpha=0.2, color=self.colors['grid'])
            axes[0].tick_params(colors='white')
        
        # Accuracy/Metric curves
        if 'train_acc' in history:
            epochs = range(1, len(history['train_acc']) + 1)
            axes[1].plot(epochs, history['train_acc'], 
                        color=self.colors['hot_pink'], linewidth=2,
                        label='Training Accuracy', marker='o', markersize=4)
            
            if 'val_acc' in history:
                axes[1].plot(epochs, history['val_acc'],
                           color=self.colors['deep_purple'], linewidth=2,
                           label='Validation Accuracy', marker='s', markersize=4)
            
            axes[1].set_xlabel('Epoch', fontsize=12, color='white')
            axes[1].set_ylabel('Accuracy', fontsize=12, color='white')
            axes[1].set_title('ACCURACY CURVES', fontsize=14,
                            color=self.colors['cyber_yellow'], fontweight='bold')
            axes[1].legend(facecolor=self.colors['grid'], edgecolor='white')
            axes[1].grid(True, alpha=0.2, color=self.colors['grid'])
            axes[1].tick_params(colors='white')
        
        plt.suptitle('TRAINING PERFORMANCE', fontsize=18,
                    color=self.colors['neon_green'], fontweight='bold')
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight',
                   facecolor=self.colors['dark_bg'])
        plt.close()
        
        return str(filepath)
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             filename: str = "confusion_matrix.png"):
        """
        Create confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class name labels
            filename: Output filename
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.colors['dark_bg'])
        ax.set_facecolor(self.colors['dark_bg'])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='plasma',
                   cbar_kws={'label': 'Count'}, ax=ax,
                   xticklabels=class_names if class_names else 'auto',
                   yticklabels=class_names if class_names else 'auto')
        
        ax.set_xlabel('Predicted Label', fontsize=12, color='white')
        ax.set_ylabel('True Label', fontsize=12, color='white')
        ax.set_title('CONFUSION MATRIX', fontsize=16,
                    color=self.colors['neon_green'], fontweight='bold', pad=20)
        ax.tick_params(colors='white')
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight',
                   facecolor=self.colors['dark_bg'])
        plt.close()
        
        return str(filepath)
    
    def plot_3d_loss_landscape(self,
                              history: Dict,
                              filename: str = "loss_landscape_3d.html"):
        """
        Create interactive 3D loss landscape using Plotly.
        
        Args:
            history: Evolution history
            filename: Output filename
        """
        generations = list(range(len(history['best_fitness'])))
        
        # Create meshgrid for surface
        gen_grid = np.array(generations)
        
        # Simulate loss surface (inverse of fitness)
        best_loss = 1 / (np.array(history['best_fitness']) + 1e-6)
        avg_loss = 1 / (np.array(history['avg_fitness']) + 1e-6)
        
        # Create 3D surface
        fig = go.Figure()
        
        # Best fitness surface
        fig.add_trace(go.Scatter3d(
            x=gen_grid,
            y=best_loss,
            z=np.zeros_like(gen_grid),
            mode='lines+markers',
            name='Best Loss',
            line=dict(color='#39FF14', width=4),
            marker=dict(size=5, color='#39FF14')
        ))
        
        # Average fitness surface
        fig.add_trace(go.Scatter3d(
            x=gen_grid,
            y=avg_loss,
            z=np.ones_like(gen_grid) * 0.5,
            mode='lines+markers',
            name='Avg Loss',
            line=dict(color='#00FFFF', width=4),
            marker=dict(size=5, color='#00FFFF')
        ))
        
        fig.update_layout(
            title='3D LOSS LANDSCAPE - EVOLUTIONARY TRAJECTORY',
            scene=dict(
                xaxis_title='Generation',
                yaxis_title='Loss',
                zaxis_title='Diversity',
                bgcolor='#0A0E27',
                xaxis=dict(gridcolor='#1A1F3A', color='white'),
                yaxis=dict(gridcolor='#1A1F3A', color='white'),
                zaxis=dict(gridcolor='#1A1F3A', color='white')
            ),
            paper_bgcolor='#0A0E27',
            plot_bgcolor='#0A0E27',
            font=dict(color='white', size=12),
            width=1200,
            height=800
        )
        
        filepath = self.output_dir / filename
        fig.write_html(filepath)
        
        return str(filepath)
    
    def plot_evolution_history(self,
                              history: Dict,
                              filename: str = "evolution_history.png"):
        """
        Plot genetic algorithm evolution statistics.
        
        Args:
            history: GA history dictionary
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                facecolor=self.colors['dark_bg'])
        
        for ax in axes:
            ax.set_facecolor(self.colors['dark_bg'])
        
        generations = range(len(history['best_fitness']))
        
        # Fitness evolution
        axes[0].plot(generations, history['best_fitness'],
                    color=self.colors['neon_green'], linewidth=3,
                    label='Best Fitness', marker='o', markersize=6)
        axes[0].plot(generations, history['avg_fitness'],
                    color=self.colors['electric_blue'], linewidth=2,
                    label='Average Fitness', marker='s', markersize=4, alpha=0.7)
        axes[0].fill_between(generations, history['best_fitness'], history['avg_fitness'],
                            color=self.colors['neon_green'], alpha=0.2)
        axes[0].set_xlabel('Generation', fontsize=12, color='white')
        axes[0].set_ylabel('Fitness', fontsize=12, color='white')
        axes[0].set_title('FITNESS EVOLUTION', fontsize=14,
                         color=self.colors['cyber_yellow'], fontweight='bold')
        axes[0].legend(facecolor=self.colors['grid'], edgecolor='white')
        axes[0].grid(True, alpha=0.2, color=self.colors['grid'])
        axes[0].tick_params(colors='white')
        
        # Diversity
        axes[1].plot(generations, history['diversity'],
                    color=self.colors['hot_pink'], linewidth=3,
                    label='Population Diversity', marker='d', markersize=6)
        axes[1].set_xlabel('Generation', fontsize=12, color='white')
        axes[1].set_ylabel('Diversity (Std Dev)', fontsize=12, color='white')
        axes[1].set_title('POPULATION DIVERSITY', fontsize=14,
                         color=self.colors['cyber_yellow'], fontweight='bold')
        axes[1].legend(facecolor=self.colors['grid'], edgecolor='white')
        axes[1].grid(True, alpha=0.2, color=self.colors['grid'])
        axes[1].tick_params(colors='white')
        
        plt.suptitle('GENETIC ALGORITHM EVOLUTION', fontsize=18,
                    color=self.colors['neon_green'], fontweight='bold')
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight',
                   facecolor=self.colors['dark_bg'])
        plt.close()
        
        return str(filepath)
    
    def generate_report(self,
                       results: Dict,
                       filename: str = "final_report.txt"):
        """
        Generate comprehensive text report.
        
        Args:
            results: Dictionary with all results
            filename: Output filename
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("  VINOGEN-CYBERCORE: NEUROEVOLUTION SYSTEM REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset info
            if 'dataset_info' in results:
                f.write("[DATASET INFORMATION]\n")
                f.write("-" * 40 + "\n")
                for key, value in results['dataset_info'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Best genome
            if 'best_genome' in results:
                genome = results['best_genome']
                f.write("[EVOLVED ARCHITECTURE]\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Hidden Layers: {' → '.join([str(n) for n in genome['hidden_layers']])}\n")
                f.write(f"  Activations: {', '.join(genome['activation_functions'])}\n")
                f.write(f"  Learning Rate: {genome['learning_rate']:.6f}\n")
                f.write(f"  Fitness Score: {genome.get('fitness', 0):.6f}\n")
                f.write(f"  Generation: {genome.get('generation', 0)}\n")
                f.write("\n")
            
            # Performance metrics
            if 'test_metrics' in results:
                f.write("[TEST SET PERFORMANCE]\n")
                f.write("-" * 40 + "\n")
                for key, value in results['test_metrics'].items():
                    f.write(f"  {key}: {value:.6f}\n")
                f.write("\n")
            
            # Classification report
            if 'classification_report' in results:
                f.write("[DETAILED CLASSIFICATION REPORT]\n")
                f.write("-" * 40 + "\n")
                f.write(results['classification_report'])
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("  End of Report\n")
            f.write("=" * 80 + "\n")
        
        return str(filepath)
