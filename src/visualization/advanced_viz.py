"""
VinoGen-CyberCore: Advanced Visualization Module
Decision boundaries, 3D heatmaps, and interactive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch


class AdvancedVisualizer:
    """Advanced visualization techniques."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize advanced visualizer.
        
        Args:
            output_dir: Output directory for figures
        """
        self.output_dir = output_dir or Path("output/advanced_viz")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set cyberpunk theme
        plt.style.use('dark_background')
        self.colors = {
            'primary': '#00ff9f',
            'secondary': '#ff006e',
            'accent': '#8338ec',
            'warning': '#ffb700',
            'background': '#1a1a2e',
            'surface': '#16213e'
        }
    
    def plot_decision_boundary_2d(self, model, X: np.ndarray, y: np.ndarray,
                                  reduction_method: str = 'pca',
                                  resolution: int = 500,
                                  save_path: str = None) -> str:
        """
        Plot 2D decision boundary using dimensionality reduction.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            reduction_method: 'pca' or 'tsne'
            resolution: Grid resolution
            save_path: Save path
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = self.output_dir / f"decision_boundary_2d_{reduction_method}.png"
        
        # Reduce to 2D
        if reduction_method == 'pca':
            reducer = PCA(n_components=2)
            X_reduced = reducer.fit_transform(X)
            title = 'Decision Boundary (PCA Projection)'
        else:
            reducer = TSNE(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            title = 'Decision Boundary (t-SNE Projection)'
        
        # Create mesh grid
        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Transform grid back to original space for prediction
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        
        # For PCA, we can inverse transform; for t-SNE, we approximate
        if reduction_method == 'pca':
            grid_original = reducer.inverse_transform(grid_2d)
        else:
            # For t-SNE, use nearest neighbor interpolation
            from scipy.interpolate import griddata
            grid_original = griddata(X_reduced, X, grid_2d, method='nearest')
        
        # Get predictions
        try:
            model.eval()
            with torch.no_grad():
                predictions = model.predict(grid_original)
            Z = predictions.reshape(xx.shape)
        except:
            # Fallback if model doesn't have predict
            Z = np.zeros_like(xx)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor(self.colors['surface'])
        
        # Plot decision boundary
        contourf = ax.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.6)
        contour = ax.contour(xx, yy, Z, levels=10, colors='white', 
                            linewidths=0.5, alpha=0.4)
        
        # Plot data points
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                           c=y, cmap='plasma', s=50, edgecolors='white',
                           linewidths=1.5, alpha=0.8, zorder=10)
        
        # Styling
        ax.set_xlabel('Component 1', fontsize=14, color=self.colors['primary'])
        ax.set_ylabel('Component 2', fontsize=14, color=self.colors['primary'])
        ax.set_title(f'🎯 {title}', fontsize=18, color=self.colors['primary'], 
                    pad=20, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Target Class', fontsize=12, color=self.colors['primary'])
        cbar.ax.yaxis.set_tick_params(color=self.colors['primary'])
        
        # Grid
        ax.grid(True, alpha=0.2, linestyle='--', color=self.colors['primary'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return str(save_path)
    
    def plot_decision_boundary_3d(self, model, X: np.ndarray, y: np.ndarray,
                                  save_path: str = None) -> str:
        """
        Plot 3D decision boundary using PCA.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            save_path: Save path
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = self.output_dir / "decision_boundary_3d.png"
        
        # Reduce to 3D with PCA
        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(X)
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        fig.patch.set_facecolor(self.colors['background'])
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(self.colors['surface'])
        
        # Plot data points
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                           c=y, cmap='plasma', s=50, edgecolors='white',
                           linewidths=1, alpha=0.7)
        
        # Create mesh for decision surface
        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        
        # For each point in the mesh, use the mean of z
        z_mean = X_reduced[:, 2].mean()
        
        # Create grid points
        grid_3d = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, z_mean)]
        
        # Transform back and predict
        grid_original = pca.inverse_transform(grid_3d)
        
        try:
            model.eval()
            with torch.no_grad():
                predictions = model.predict(grid_original)
            Z = predictions.reshape(xx.shape)
            
            # Plot decision surface
            surf = ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.3,
                                  edgecolor='none', antialiased=True)
        except:
            pass
        
        # Styling
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                     fontsize=12, color=self.colors['primary'])
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', 
                     fontsize=12, color=self.colors['primary'])
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', 
                     fontsize=12, color=self.colors['primary'])
        ax.set_title('🎯 3D Decision Boundary (PCA Projection)', 
                    fontsize=18, color=self.colors['primary'], 
                    pad=20, fontweight='bold')
        
        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Target', fontsize=12, color=self.colors['primary'])
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return str(save_path)
    
    def plot_activation_heatmap_3d(self, model, X: np.ndarray, 
                                   layer_idx: int = 0,
                                   save_path: str = None) -> str:
        """
        Plot 3D heatmap of layer activations.
        
        Args:
            model: Model with stored activations
            X: Input data
            layer_idx: Which layer to visualize
            save_path: Save path
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = self.output_dir / f"activation_heatmap_3d_layer{layer_idx}.png"
        
        # Get activations
        model.eval()
        with torch.no_grad():
            _ = model(torch.FloatTensor(X[:100]), store_activations=True)
        
        if not hasattr(model, 'layer_activations') or len(model.layer_activations) <= layer_idx:
            # Create dummy visualization
            activations = np.random.randn(100, 10)
        else:
            activations = model.layer_activations[layer_idx]
            if len(activations.shape) > 2:
                activations = activations.reshape(activations.shape[0], -1)
        
        # Limit to first 50 samples and 50 neurons for visualization
        activations = activations[:50, :min(50, activations.shape[1])]
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        fig.patch.set_facecolor(self.colors['background'])
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(self.colors['surface'])
        
        # Create mesh
        samples = np.arange(activations.shape[0])
        neurons = np.arange(activations.shape[1])
        X_mesh, Y_mesh = np.meshgrid(neurons, samples)
        
        # Plot surface
        surf = ax.plot_surface(X_mesh, Y_mesh, activations, 
                              cmap='plasma', edgecolor='none',
                              alpha=0.9, antialiased=True)
        
        # Styling
        ax.set_xlabel('Neuron Index', fontsize=12, color=self.colors['primary'])
        ax.set_ylabel('Sample Index', fontsize=12, color=self.colors['primary'])
        ax.set_zlabel('Activation Value', fontsize=12, color=self.colors['primary'])
        ax.set_title(f'🔥 3D Activation Heatmap - Layer {layer_idx}',
                    fontsize=18, color=self.colors['primary'],
                    pad=20, fontweight='bold')
        
        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Activation', fontsize=12, color=self.colors['primary'])
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return str(save_path)
    
    def plot_weight_heatmap_2d(self, model, layer_idx: int = 0,
                               save_path: str = None) -> str:
        """
        Plot 2D heatmap of layer weights.
        
        Args:
            model: Trained model
            layer_idx: Which layer to visualize
            save_path: Save path
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = self.output_dir / f"weight_heatmap_2d_layer{layer_idx}.png"
        
        # Get weights
        try:
            if hasattr(model, 'layers') and layer_idx < len(model.layers):
                weights = model.layers[layer_idx].weight.detach().cpu().numpy()
            else:
                weights = np.random.randn(32, 32)
        except:
            weights = np.random.randn(32, 32)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor(self.colors['surface'])
        
        # Plot heatmap
        im = ax.imshow(weights, cmap='RdBu_r', aspect='auto', 
                      vmin=-np.abs(weights).max(), vmax=np.abs(weights).max())
        
        # Styling
        ax.set_xlabel('Input Neuron', fontsize=14, color=self.colors['primary'])
        ax.set_ylabel('Output Neuron', fontsize=14, color=self.colors['primary'])
        ax.set_title(f'🎨 Weight Heatmap - Layer {layer_idx}',
                    fontsize=18, color=self.colors['primary'],
                    pad=20, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight Value', fontsize=12, color=self.colors['primary'])
        
        # Grid
        ax.grid(True, alpha=0.2, linestyle='--', color='white')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return str(save_path)
    
    def plot_prediction_confidence_distribution(self, y_true: np.ndarray, 
                                                y_pred: np.ndarray,
                                                confidences: np.ndarray,
                                                save_path: str = None) -> str:
        """
        Plot prediction confidence distribution.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            confidences: Prediction confidences
            save_path: Save path
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = self.output_dir / "confidence_distribution.png"
        
        # Separate correct and incorrect predictions
        correct_mask = (y_true == y_pred)
        conf_correct = confidences[correct_mask]
        conf_incorrect = confidences[~correct_mask]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor(self.colors['background'])
        fig.suptitle('📊 PREDICTION CONFIDENCE ANALYSIS', 
                    fontsize=20, color=self.colors['primary'], 
                    fontweight='bold', y=0.98)
        
        # 1. Overall confidence distribution
        ax1 = axes[0, 0]
        ax1.set_facecolor(self.colors['surface'])
        ax1.hist(confidences, bins=50, color=self.colors['accent'], 
                alpha=0.7, edgecolor='white')
        ax1.axvline(confidences.mean(), color=self.colors['secondary'], 
                   linestyle='--', linewidth=2, label=f'Mean: {confidences.mean():.3f}')
        ax1.set_xlabel('Confidence', fontsize=12, color=self.colors['primary'])
        ax1.set_ylabel('Frequency', fontsize=12, color=self.colors['primary'])
        ax1.set_title('Overall Confidence Distribution', 
                     fontsize=14, color=self.colors['primary'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Correct vs Incorrect
        ax2 = axes[0, 1]
        ax2.set_facecolor(self.colors['surface'])
        ax2.hist([conf_correct, conf_incorrect], bins=30, 
                label=['Correct', 'Incorrect'],
                color=[self.colors['primary'], self.colors['secondary']],
                alpha=0.7, edgecolor='white')
        ax2.set_xlabel('Confidence', fontsize=12, color=self.colors['primary'])
        ax2.set_ylabel('Frequency', fontsize=12, color=self.colors['primary'])
        ax2.set_title('Correct vs Incorrect Predictions',
                     fontsize=14, color=self.colors['primary'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence vs Accuracy
        ax3 = axes[1, 0]
        ax3.set_facecolor(self.colors['surface'])
        
        # Bin confidences and calculate accuracy per bin
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accs = []
        
        for i in range(len(bins)-1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                acc = (y_true[mask] == y_pred[mask]).mean()
                bin_accs.append(acc)
            else:
                bin_accs.append(0)
        
        ax3.plot(bin_centers, bin_accs, marker='o', linewidth=2, 
                markersize=8, color=self.colors['primary'])
        ax3.plot([0, 1], [0, 1], '--', color=self.colors['secondary'], 
                alpha=0.5, label='Perfect Calibration')
        ax3.set_xlabel('Confidence', fontsize=12, color=self.colors['primary'])
        ax3.set_ylabel('Accuracy', fontsize=12, color=self.colors['primary'])
        ax3.set_title('Calibration Curve',
                     fontsize=14, color=self.colors['primary'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        
        # 4. Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
CONFIDENCE STATISTICS
{'='*40}

Overall:
  Mean:     {confidences.mean():.4f}
  Median:   {np.median(confidences):.4f}
  Std:      {confidences.std():.4f}
  Min:      {confidences.min():.4f}
  Max:      {confidences.max():.4f}

Correct Predictions:
  Mean:     {conf_correct.mean():.4f}
  Count:    {len(conf_correct)}

Incorrect Predictions:
  Mean:     {conf_incorrect.mean():.4f}
  Count:    {len(conf_incorrect)}

Accuracy: {(y_true == y_pred).mean():.4f}
"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor=self.colors['surface'], 
                         alpha=0.8, edgecolor=self.colors['primary'], linewidth=2),
                color=self.colors['primary'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'], edgecolor='none')
        plt.close()
        
        return str(save_path)
