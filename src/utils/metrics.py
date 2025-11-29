"""
VinoGen-CyberCore: Advanced Metrics Module
Comprehensive metrics for both classification and regression tasks.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, Tuple, Optional
import time


class MetricsTracker:
    """Advanced metrics tracking for model evaluation."""
    
    def __init__(self, task: str = "classification"):
        """
        Initialize metrics tracker.
        
        Args:
            task: Type of task ('classification' or 'regression')
        """
        self.task = task
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.training_times = []
        self.epoch_times = []
        self.losses = []
        
    def add_batch(self, y_pred: np.ndarray, y_true: np.ndarray, 
                  loss: float = None, confidence: np.ndarray = None,
                  batch_time: float = None):
        """
        Add batch results for tracking.
        
        Args:
            y_pred: Predictions
            y_true: True labels/values
            loss: Batch loss
            confidence: Prediction confidences
            batch_time: Time taken for batch
        """
        self.predictions.extend(y_pred.flatten())
        self.targets.extend(y_true.flatten())
        
        if loss is not None:
            self.losses.append(loss)
        
        if confidence is not None:
            self.confidences.extend(confidence.flatten())
        
        if batch_time is not None:
            self.epoch_times.append(batch_time)
    
    def compute_classification_metrics(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray) -> Dict:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle multi-class
        average_method = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
        
        metrics['precision'] = precision_score(y_true, y_pred, 
                                              average=average_method, 
                                              zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, 
                                        average=average_method,
                                        zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, 
                                      average=average_method,
                                      zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        try:
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics['per_class'] = report
        except:
            metrics['per_class'] = {}
        
        # Error analysis
        metrics['total_samples'] = len(y_true)
        metrics['correct_predictions'] = np.sum(y_true == y_pred)
        metrics['incorrect_predictions'] = np.sum(y_true != y_pred)
        metrics['error_rate'] = 1 - metrics['accuracy']
        
        return metrics
    
    def compute_regression_metrics(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict:
        """
        Compute comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic error metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            metrics['mape'] = 0.0
        
        # Error statistics
        errors = y_true - y_pred
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['max_error'] = np.max(np.abs(errors))
        metrics['min_error'] = np.min(np.abs(errors))
        
        # Quantile errors
        metrics['q25_error'] = np.percentile(np.abs(errors), 25)
        metrics['q50_error'] = np.percentile(np.abs(errors), 50)  # Median
        metrics['q75_error'] = np.percentile(np.abs(errors), 75)
        
        # Range analysis
        metrics['prediction_range'] = (np.min(y_pred), np.max(y_pred))
        metrics['true_range'] = (np.min(y_true), np.max(y_true))
        
        # Explained variance
        metrics['explained_variance'] = 1 - (np.var(errors) / np.var(y_true))
        
        metrics['total_samples'] = len(y_true)
        
        return metrics
    
    def compute_all_metrics(self) -> Dict:
        """
        Compute all available metrics based on accumulated data.
        
        Returns:
            Complete metrics dictionary
        """
        if len(self.predictions) == 0:
            return {}
        
        y_pred = np.array(self.predictions)
        y_true = np.array(self.targets)
        
        metrics = {
            'task': self.task,
            'n_samples': len(y_pred)
        }
        
        # Compute task-specific metrics
        if self.task == 'classification':
            task_metrics = self.compute_classification_metrics(y_true, y_pred)
        else:
            task_metrics = self.compute_regression_metrics(y_true, y_pred)
        
        metrics.update(task_metrics)
        
        # Add timing information
        if self.training_times:
            metrics['total_training_time'] = sum(self.training_times)
            metrics['avg_epoch_time'] = np.mean(self.training_times)
        
        if self.epoch_times:
            metrics['avg_batch_time'] = np.mean(self.epoch_times)
        
        # Add loss information
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
            metrics['final_loss'] = self.losses[-1] if self.losses else 0.0
        
        # Add confidence information (if available)
        if self.confidences:
            metrics['avg_confidence'] = np.mean(self.confidences)
            metrics['min_confidence'] = np.min(self.confidences)
            metrics['max_confidence'] = np.max(self.confidences)
        
        return metrics
    
    def get_summary_string(self, metrics: Dict = None) -> str:
        """
        Get a formatted string summary of metrics.
        
        Args:
            metrics: Metrics dictionary (if None, computes from accumulated data)
            
        Returns:
            Formatted string
        """
        if metrics is None:
            metrics = self.compute_all_metrics()
        
        if not metrics:
            return "No metrics available"
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"METRICS SUMMARY ({metrics['task'].upper()})")
        lines.append("=" * 60)
        
        if metrics['task'] == 'classification':
            lines.append(f"Accuracy:     {metrics.get('accuracy', 0):.4f}")
            lines.append(f"Precision:    {metrics.get('precision', 0):.4f}")
            lines.append(f"Recall:       {metrics.get('recall', 0):.4f}")
            lines.append(f"F1-Score:     {metrics.get('f1_score', 0):.4f}")
            lines.append(f"Error Rate:   {metrics.get('error_rate', 0):.4f}")
            lines.append(f"Correct:      {metrics.get('correct_predictions', 0)}/{metrics.get('total_samples', 0)}")
        else:
            lines.append(f"MAE:          {metrics.get('mae', 0):.4f}")
            lines.append(f"RMSE:         {metrics.get('rmse', 0):.4f}")
            lines.append(f"R² Score:     {metrics.get('r2_score', 0):.4f}")
            lines.append(f"MAPE:         {metrics.get('mape', 0):.2f}%")
            lines.append(f"Max Error:    {metrics.get('max_error', 0):.4f}")
            lines.append(f"Mean Error:   {metrics.get('mean_error', 0):.4f} ± {metrics.get('std_error', 0):.4f}")
        
        if 'avg_epoch_time' in metrics:
            lines.append(f"\nAvg Epoch Time: {metrics['avg_epoch_time']:.2f}s")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class PerformanceTimer:
    """Track performance and timing metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset timer."""
        self.start_time = None
        self.end_time = None
        self.epoch_times = []
        self.batch_times = []
        self.phase_times = {}
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing."""
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def record_epoch(self, epoch_time: float):
        """Record an epoch time."""
        self.epoch_times.append(epoch_time)
    
    def record_batch(self, batch_time: float):
        """Record a batch time."""
        self.batch_times.append(batch_time)
    
    def record_phase(self, phase_name: str, phase_time: float):
        """Record a phase time."""
        self.phase_times[phase_name] = phase_time
    
    def get_statistics(self) -> Dict:
        """Get timing statistics."""
        stats = {
            'total_time': self.elapsed(),
            'n_epochs': len(self.epoch_times),
            'n_batches': len(self.batch_times)
        }
        
        if self.epoch_times:
            stats['avg_epoch_time'] = np.mean(self.epoch_times)
            stats['total_epoch_time'] = np.sum(self.epoch_times)
            stats['min_epoch_time'] = np.min(self.epoch_times)
            stats['max_epoch_time'] = np.max(self.epoch_times)
        
        if self.batch_times:
            stats['avg_batch_time'] = np.mean(self.batch_times)
            stats['total_batch_time'] = np.sum(self.batch_times)
        
        stats['phases'] = self.phase_times.copy()
        
        return stats
