"""
VinoGen-CyberCore: Dynamic MLP Model with PyTorch
Supports variable architectures evolved by Genetic Algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class DynamicMLP(nn.Module):
    """
    A Multi-Layer Perceptron with dynamic architecture.
    
    The network structure is defined by a genome:
    - Number of hidden layers
    - Neurons per layer
    - Activation functions per layer
    - Learning rate
    
    Mathematical Foundation:
    ========================
    
    Forward Pass:
    -------------
    For layer l: z[l] = W[l] @ a[l-1] + b[l]
                 a[l] = σ(z[l])
    
    Where:
    - W[l]: Weight matrix for layer l
    - b[l]: Bias vector for layer l
    - σ: Activation function (ReLU, Sigmoid, Tanh, etc.)
    - a[l]: Activation output of layer l
    
    Loss Functions:
    ---------------
    Classification (Cross-Entropy):
        L = -Σ y_true * log(y_pred)
    
    Regression (MSE):
        L = (1/n) * Σ(y_true - y_pred)²
    
    Backpropagation:
    ----------------
    ∂L/∂W[l] = ∂L/∂a[l] * ∂a[l]/∂z[l] * ∂z[l]/∂W[l]
    
    Gradient Descent:
    -----------------
    W[l] ← W[l] - α * ∂L/∂W[l]
    
    Where α is the learning rate.
    """
    
    def __init__(self, genome: Dict, input_dim: int, output_dim: int, task: str = "classification"):
        """
        Initialize the neural network with evolved genome.
        
        Args:
            genome: Dictionary containing architecture genes
            input_dim: Number of input features
            output_dim: Number of output classes (classification) or 1 (regression)
            task: "classification" or "regression"
        """
        super(DynamicMLP, self).__init__()
        
        self.genome = genome
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.activations = []
        
        # Parse genome - handle both Genome objects and dicts
        if hasattr(genome, 'hidden_layers'):
            # It's a Genome object
            hidden_layers = genome.hidden_layers
            activation_funcs = genome.activation_functions
            learning_rate = genome.learning_rate
        else:
            # It's a dict
            hidden_layers = genome['hidden_layers']
            activation_funcs = genome['activation_functions']
            learning_rate = genome['learning_rate']
        
        # Build architecture
        prev_dim = input_dim
        
        for i, (n_neurons, act_name) in enumerate(zip(hidden_layers, activation_funcs)):
            # Add linear layer
            self.layers.append(nn.Linear(prev_dim, n_neurons))
            # Store activation function
            self.activations.append(self._get_activation(act_name))
            prev_dim = n_neurons
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Loss function
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        
        # Optimizer (learning rate from genome)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Training history
        self.history = defaultdict(list)
        
        # Activation storage for visualization
        self.layer_activations = []
    
    def _get_activation(self, name: str) -> nn.Module:
        """Map activation name to PyTorch module."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor, store_activations: bool = False) -> torch.Tensor:
        """
        Forward propagation through the network.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            store_activations: Whether to store intermediate activations
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        if store_activations:
            self.layer_activations = [x.detach().cpu().numpy()]
        
        # Pass through hidden layers
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            x = activation(x)
            
            if store_activations:
                self.layer_activations.append(x.detach().cpu().numpy())
        
        # Output layer
        x = self.output_layer(x)
        
        if store_activations:
            self.layer_activations.append(x.detach().cpu().numpy())
        
        return x
    
    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray, 
                   batch_size: int = 32) -> float:
        """
        Train for one epoch.
        
        Args:
            X_train: Training features
            y_train: Training labels
            batch_size: Mini-batch size
            
        Returns:
            Average loss for the epoch
        """
        self.train()
        total_loss = 0.0
        n_batches = 0
        
        # Create mini-batches
        indices = np.random.permutation(len(X_train))
        
        for start_idx in range(0, len(X_train), batch_size):
            end_idx = min(start_idx + batch_size, len(X_train))
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch
            X_batch = torch.FloatTensor(X_train[batch_indices])
            y_batch = torch.LongTensor(y_train[batch_indices]) if self.task == "classification" \
                      else torch.FloatTensor(y_train[batch_indices]).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.forward(X_batch)
            
            # Compute loss
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Dictionary with metrics
        """
        self.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y) if self.task == "classification" \
                       else torch.FloatTensor(y).unsqueeze(1)
            
            outputs = self.forward(X_tensor)
            loss = self.criterion(outputs, y_tensor).item()
            
            if self.task == "classification":
                # Classification metrics
                predictions = torch.argmax(outputs, dim=1).numpy()
                accuracy = np.mean(predictions == y)
                return {'loss': loss, 'accuracy': accuracy}
            else:
                # Regression metrics
                predictions = outputs.squeeze().numpy()
                mae = np.mean(np.abs(predictions - y))
                rmse = np.sqrt(np.mean((predictions - y) ** 2))
                r2 = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
                return {'loss': loss, 'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        self.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.forward(X_tensor)
            
            if self.task == "classification":
                predictions = torch.argmax(outputs, dim=1).numpy()
            else:
                predictions = outputs.squeeze().numpy()
            
            return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        
        self.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.forward(X_tensor)
            probas = torch.softmax(outputs, dim=1).numpy()
            
            return probas
    
    def get_activations(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Get layer activations for visualization.
        
        Args:
            X: Single input sample [input_dim]
            
        Returns:
            List of activation arrays for each layer
        """
        self.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
            _ = self.forward(X_tensor, store_activations=True)
            
            return self.layer_activations
    
    def get_architecture_summary(self) -> str:
        """
        Generate a human-readable summary of the network architecture.
        
        Returns formatted string with layer details and parameter counts.
        """
        summary = []
        summary.append("\n╔══════════════════════════════════════════════════════════════╗")
        summary.append("║           NEURAL NETWORK ARCHITECTURE                       ║")
        summary.append("╚══════════════════════════════════════════════════════════════╝")
        
        total_params = 0
        
        # Input layer
        summary.append(f"\n[INPUT LAYER]  Dimensions: {self.input_dim}")
        
        # Hidden layers
        prev_dim = self.input_dim
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            n_params = layer.weight.numel() + layer.bias.numel()
            total_params += n_params
            summary.append(f"\n[HIDDEN {i+1}]    Neurons: {layer.out_features}")
            summary.append(f"              Activation: {activation.__class__.__name__}")
            summary.append(f"              Parameters: {n_params:,}")
            summary.append(f"              Shape: ({layer.in_features}, {layer.out_features})")
            prev_dim = layer.out_features
        
        # Output layer
        n_params = self.output_layer.weight.numel() + self.output_layer.bias.numel()
        total_params += n_params
        summary.append(f"\n[OUTPUT]      Neurons: {self.output_dim}")
        summary.append(f"              Task: {self.task.upper()}")
        summary.append(f"              Parameters: {n_params:,}")
        
        # Summary
        summary.append(f"\n{'─' * 62}")
        summary.append(f"TOTAL PARAMETERS: {total_params:,}")
        summary.append(f"LEARNING RATE:    {self.learning_rate:.6f}")
        summary.append(f"OPTIMIZER:        Adam")
        summary.append(f"{'─' * 62}\n")
        
        return "\n".join(summary)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
