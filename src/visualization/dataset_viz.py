"""
VinoGen-CyberCore: Dataset Analysis Module
Comprehensive dataset exploration and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DatasetAnalyzer:
    """Advanced dataset analysis and visualization."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize dataset analyzer.
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = output_dir or Path("output/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("darkgrid")
        plt.rcParams['figure.facecolor'] = '#1a1a2e'
        plt.rcParams['axes.facecolor'] = '#16213e'
        plt.rcParams['text.color'] = '#eee'
        plt.rcParams['axes.labelcolor'] = '#eee'
        plt.rcParams['xtick.color'] = '#eee'
        plt.rcParams['ytick.color'] = '#eee'
        plt.rcParams['grid.color'] = '#2a2a4e'
    
    def analyze_dataset(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str] = None) -> Dict:
        """
        Perform comprehensive dataset analysis.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: Names of features
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['n_samples'] = X.shape[0]
        analysis['n_features'] = X.shape[1]
        analysis['n_classes'] = len(np.unique(y))
        
        # Feature statistics
        analysis['feature_stats'] = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0),
            'median': np.median(X, axis=0),
            'q25': np.percentile(X, 25, axis=0),
            'q75': np.percentile(X, 75, axis=0)
        }
        
        # Target statistics
        analysis['target_stats'] = {
            'unique_values': np.unique(y),
            'value_counts': np.bincount(y.astype(int)) if y.dtype == int else None,
            'mean': np.mean(y),
            'std': np.std(y),
            'min': np.min(y),
            'max': np.max(y)
        }
        
        # Class balance (for classification)
        if len(np.unique(y)) < 20:  # Assume classification
            unique, counts = np.unique(y, return_counts=True)
            analysis['class_distribution'] = dict(zip(unique.astype(int), counts))
            analysis['class_balance'] = counts / len(y)
            analysis['is_balanced'] = (np.max(counts) / np.min(counts)) < 2.0
        
        # Missing values
        analysis['missing_values'] = np.sum(np.isnan(X), axis=0)
        analysis['has_missing'] = np.any(np.isnan(X))
        
        # Correlation analysis
        if feature_names and len(feature_names) == X.shape[1]:
            df = pd.DataFrame(X, columns=feature_names)
            analysis['correlation_matrix'] = df.corr().values
            analysis['feature_names'] = feature_names
        else:
            analysis['correlation_matrix'] = np.corrcoef(X.T)
            analysis['feature_names'] = [f'Feature {i}' for i in range(X.shape[1])]
        
        # Outlier detection (IQR method)
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).sum(axis=0)
        analysis['outliers_per_feature'] = outliers
        analysis['total_outliers'] = outliers.sum()
        
        return analysis
    
    def plot_dataset_overview(self, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str] = None,
                             save_path: str = None) -> str:
        """
        Create comprehensive dataset overview visualization.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: Feature names
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = self.output_dir / "dataset_overview.png"
        
        # Perform analysis
        analysis = self.analyze_dataset(X, y, feature_names)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('📊 DATASET OVERVIEW', fontsize=24, 
                    color='#00ff9f', fontweight='bold', y=0.98)
        
        # 1. Dataset Statistics (text box)
        ax1 = plt.subplot(3, 3, 1)
        ax1.axis('off')
        stats_text = f"""
DATASET STATISTICS
{'='*35}
Samples:        {analysis['n_samples']:,}
Features:       {analysis['n_features']}
Classes:        {analysis['n_classes']}
Missing Values: {analysis['total_outliers']}

TARGET DISTRIBUTION
{'='*35}
Mean:   {analysis['target_stats']['mean']:.3f}
Std:    {analysis['target_stats']['std']:.3f}
Min:    {analysis['target_stats']['min']:.3f}
Max:    {analysis['target_stats']['max']:.3f}
"""
        ax1.text(0.1, 0.9, stats_text, transform=ax1.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8),
                color='#00ff9f')
        
        # 2. Class Distribution
        ax2 = plt.subplot(3, 3, 2)
        if 'class_distribution' in analysis:
            classes = list(analysis['class_distribution'].keys())
            counts = list(analysis['class_distribution'].values())
            colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
            ax2.bar(classes, counts, color=colors, edgecolor='white', linewidth=1.5)
            ax2.set_xlabel('Class', fontsize=12, color='#00ff9f')
            ax2.set_ylabel('Count', fontsize=12, color='#00ff9f')
            ax2.set_title('Class Distribution', fontsize=14, color='#00ff9f', pad=10)
            ax2.grid(True, alpha=0.3)
        
        # 3. Target Distribution
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(y, bins=30, color='#ff006e', alpha=0.7, edgecolor='white')
        ax3.set_xlabel('Target Value', fontsize=12, color='#00ff9f')
        ax3.set_ylabel('Frequency', fontsize=12, color='#00ff9f')
        ax3.set_title('Target Value Distribution', fontsize=14, color='#00ff9f', pad=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Correlation Heatmap
        ax4 = plt.subplot(3, 3, 4)
        corr_matrix = analysis['correlation_matrix']
        im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_title('Feature Correlations', fontsize=14, color='#00ff9f', pad=10)
        plt.colorbar(im, ax=ax4)
        
        # 5. Feature Distributions (Box Plot)
        ax5 = plt.subplot(3, 3, 5)
        # Normalize features for better visualization
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        bp = ax5.boxplot(X_norm, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#8338ec')
            patch.set_alpha(0.7)
        ax5.set_xlabel('Feature Index', fontsize=12, color='#00ff9f')
        ax5.set_ylabel('Normalized Value', fontsize=12, color='#00ff9f')
        ax5.set_title('Feature Distributions (Normalized)', fontsize=14, color='#00ff9f', pad=10)
        ax5.grid(True, alpha=0.3)
        
        # 6. PCA Visualization (2D)
        ax6 = plt.subplot(3, 3, 6)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                             s=20, alpha=0.6, edgecolors='white', linewidths=0.5)
        ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', 
                      fontsize=12, color='#00ff9f')
        ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', 
                      fontsize=12, color='#00ff9f')
        ax6.set_title('PCA Projection (2D)', fontsize=14, color='#00ff9f', pad=10)
        plt.colorbar(scatter, ax=ax6, label='Target')
        ax6.grid(True, alpha=0.3)
        
        # 7. Feature Statistics
        ax7 = plt.subplot(3, 3, 7)
        feature_means = analysis['feature_stats']['mean']
        feature_stds = analysis['feature_stats']['std']
        x_pos = np.arange(len(feature_means))
        ax7.bar(x_pos, feature_means, yerr=feature_stds, 
               color='#06ffa5', alpha=0.7, edgecolor='white', linewidth=1.5,
               error_kw={'elinewidth': 1, 'ecolor': '#ff006e'})
        ax7.set_xlabel('Feature Index', fontsize=12, color='#00ff9f')
        ax7.set_ylabel('Mean ± Std', fontsize=12, color='#00ff9f')
        ax7.set_title('Feature Statistics', fontsize=14, color='#00ff9f', pad=10)
        ax7.grid(True, alpha=0.3)
        
        # 8. Outliers Detection
        ax8 = plt.subplot(3, 3, 8)
        outliers = analysis['outliers_per_feature']
        ax8.bar(range(len(outliers)), outliers, color='#ff006e', 
               alpha=0.7, edgecolor='white', linewidth=1.5)
        ax8.set_xlabel('Feature Index', fontsize=12, color='#00ff9f')
        ax8.set_ylabel('Number of Outliers', fontsize=12, color='#00ff9f')
        ax8.set_title('Outlier Detection (IQR Method)', fontsize=14, color='#00ff9f', pad=10)
        ax8.grid(True, alpha=0.3)
        
        # 9. Data Quality Summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        quality_score = 100
        if analysis['has_missing']:
            quality_score -= 20
        if not analysis.get('is_balanced', True):
            quality_score -= 15
        if analysis['total_outliers'] > analysis['n_samples'] * 0.05:
            quality_score -= 15
        
        quality_text = f"""
DATA QUALITY REPORT
{'='*35}
Overall Score:  {quality_score}/100

✓ Missing Values: {'None' if not analysis['has_missing'] else 'Present'}
✓ Class Balance:  {'Good' if analysis.get('is_balanced', True) else 'Imbalanced'}
✓ Outliers:       {analysis['total_outliers']} detected
✓ Features:       {analysis['n_features']} total

RECOMMENDATIONS:
{'='*35}
"""
        if not analysis.get('is_balanced', True):
            quality_text += "• Consider class rebalancing\n"
        if analysis['total_outliers'] > 0:
            quality_text += "• Review outliers\n"
        if quality_score > 85:
            quality_text += "• Dataset is high quality! ✓\n"
        
        color = '#00ff9f' if quality_score > 85 else '#ffb700' if quality_score > 70 else '#ff006e'
        ax9.text(0.1, 0.9, quality_text, transform=ax9.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8),
                color=color)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='#1a1a2e', edgecolor='none')
        plt.close()
        
        return str(save_path)
    
    def plot_sample_visualization(self, X: np.ndarray, y: np.ndarray,
                                 n_samples: int = 10,
                                 feature_names: List[str] = None,
                                 save_path: str = None) -> str:
        """
        Visualize random data samples.
        
        Args:
            X: Feature matrix
            y: Target values
            n_samples: Number of samples to show
            feature_names: Feature names
            save_path: Save path
            
        Returns:
            Path to saved figure
        """
        if save_path is None:
            save_path = self.output_dir / "sample_visualization.png"
        
        # Select random samples
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X_samples = X[indices]
        y_samples = y[indices]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('📋 DATASET SAMPLES', fontsize=24, 
                    color='#00ff9f', fontweight='bold')
        
        for idx, (ax, sample, target) in enumerate(zip(axes.flat, X_samples, y_samples)):
            # Plot feature values
            feature_indices = range(len(sample))
            colors = plt.cm.viridis(np.linspace(0, 1, len(sample)))
            ax.bar(feature_indices, sample, color=colors, edgecolor='white', linewidth=1)
            ax.set_title(f'Sample {idx+1} | Target: {target:.2f}', 
                        fontsize=12, color='#00ff9f', pad=5)
            ax.set_xlabel('Feature', fontsize=10, color='#aaa')
            ax.set_ylabel('Value', fontsize=10, color='#aaa')
            ax.grid(True, alpha=0.3)
            
            # Add feature names if available
            if feature_names and len(feature_names) <= 15:
                ax.set_xticks(feature_indices)
                ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='#1a1a2e', edgecolor='none')
        plt.close()
        
        return str(save_path)
