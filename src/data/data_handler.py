"""
VinoGen-CyberCore: Data Handler Module
Handles wine dataset loading, preprocessing, and synthetic data generation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataHandler:
    """
    Manages wine quality dataset with cyberpunk flair.
    
    Capabilities:
    - Load real wine datasets (CSV)
    - Generate synthetic wine data
    - Preprocess and normalize features
    - Split data for training/validation/testing
    """
    
    def __init__(self, data_path: Optional[str] = None, task: str = "classification"):
        """
        Initialize the Data Matrix.
        
        Args:
            data_path: Path to wine dataset CSV
            task: "classification" or "regression"
        """
        self.data_path = data_path
        self.task = task
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Metadata
        self.feature_names = None
        self.n_features = None
        self.n_classes = None
        self.dataset_info = {}
        
    def load_data(self) -> bool:
        """
        Load wine dataset from file or generate synthetic data.
        
        Returns:
            Success status
        """
        try:
            if self.data_path and Path(self.data_path).exists():
                df = pd.read_csv(self.data_path)
                return self._process_real_data(df)
            else:
                return self._generate_synthetic_data()
        except Exception as e:
            print(f"[ERROR] Data loading failed: {e}")
            return False
    
    def _process_real_data(self, df: pd.DataFrame) -> bool:
        """Process real wine dataset."""
        # Identify target column (usually 'quality' or 'type')
        possible_targets = ['quality', 'type', 'label', 'class']
        target_col = None
        
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # Assume last column is target
            target_col = df.columns[-1]
        
        # Separate features and target
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != target_col]
        self.n_features = X.shape[1]
        
        # Process target based on task
        if self.task == "classification":
            # Convert to categorical and ensure 0-based indexing
            if y.dtype in [np.float32, np.float64]:
                # If continuous, bin into classes
                y = pd.qcut(y, q=min(10, len(np.unique(y))), labels=False, duplicates='drop')
            
            # Remap to 0-based labels for PyTorch
            unique_labels = np.unique(y)
            label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
            y = np.array([label_map[label] for label in y])
            
            self.n_classes = len(np.unique(y))
        else:
            # Regression: keep continuous
            self.n_classes = 1
            y = y.astype(np.float32)
        
        # Split data
        self._split_data(X, y)
        
        # Store metadata
        self.dataset_info = {
            'source': 'real',
            'samples': len(X),
            'features': self.n_features,
            'classes': self.n_classes,
            'task': self.task
        }
        
        return True
    
    def _generate_synthetic_data(self, n_samples: int = 5000) -> bool:
        """
        Generate synthetic wine quality data with realistic distributions.
        
        Based on typical wine chemistry:
        - Fixed acidity: 4-16 g/dm³
        - Volatile acidity: 0.1-1.6 g/dm³
        - Citric acid: 0-1 g/dm³
        - Residual sugar: 0.9-15.5 g/dm³
        - Chlorides: 0.01-0.6 g/dm³
        - Free sulfur dioxide: 1-72 mg/dm³
        - Total sulfur dioxide: 6-289 mg/dm³
        - Density: 0.990-1.004 g/cm³
        - pH: 2.74-4.01
        - Sulphates: 0.33-2.00 g/dm³
        - Alcohol: 8-15% vol
        """
        np.random.seed(42)
        
        # Generate features with correlations
        fixed_acidity = np.random.normal(8.3, 1.7, n_samples)
        volatile_acidity = np.random.gamma(2, 0.2, n_samples)
        citric_acid = np.random.beta(2, 5, n_samples)
        residual_sugar = np.random.gamma(2, 2, n_samples)
        chlorides = np.random.gamma(2, 0.04, n_samples)
        free_sulfur = np.random.gamma(3, 10, n_samples)
        total_sulfur = free_sulfur + np.random.gamma(4, 20, n_samples)
        density = 0.996 + np.random.normal(0, 0.002, n_samples)
        pH = 3.3 + np.random.normal(0, 0.15, n_samples)
        sulphates = np.random.gamma(3, 0.2, n_samples)
        alcohol = np.random.normal(10.5, 1.2, n_samples)
        
        # Stack features
        X = np.column_stack([
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur,
            total_sulfur,
            density,
            pH,
            sulphates,
            alcohol
        ])
        
        # Generate quality scores based on feature relationships
        # Quality = f(alcohol, acidity, sulfur, etc.)
        quality_score = (
            0.3 * (alcohol - 8) / 7 +  # Higher alcohol = better
            0.2 * (1 - volatile_acidity) +  # Lower volatile acidity = better
            0.15 * citric_acid +  # More citric acid = better
            0.15 * (1 - density) * 100 +  # Lower density = better
            0.1 * sulphates +  # More sulphates = better
            0.1 * np.random.randn(n_samples) * 0.5  # Random noise
        )
        
        if self.task == "classification":
            # Convert to quality classes (3-9)
            y_raw = np.clip(np.round(quality_score * 3 + 6), 3, 9).astype(int)
            
            # Remap to 0-based labels for PyTorch
            # Original: 3,4,5,6,7,8,9 -> New: 0,1,2,3,4,5,6
            y = y_raw - y_raw.min()
            self.n_classes = len(np.unique(y))
        else:
            # Regression: continuous quality
            y = np.clip(quality_score * 3 + 6, 0, 10)
            self.n_classes = 1
        
        # Feature names
        self.feature_names = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 
            'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
            'total_sulfur_dioxide', 'density', 'pH', 
            'sulphates', 'alcohol'
        ]
        self.n_features = len(self.feature_names)
        
        # Save synthetic data to CSV in input folder
        self._save_synthetic_to_csv(X, y if self.task == "classification" else y)
        
        # Split data
        self._split_data(X, y)
        
        # Store metadata
        self.dataset_info = {
            'source': 'synthetic',
            'samples': n_samples,
            'features': self.n_features,
            'classes': self.n_classes,
            'task': self.task
        }
        
        return True
    
    def _save_synthetic_to_csv(self, X: np.ndarray, y: np.ndarray):
        """Save generated synthetic data to CSV file in input folder."""
        try:
            # Create input directory if it doesn't exist
            input_dir = Path('input')
            input_dir.mkdir(exist_ok=True, parents=True)
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=self.feature_names)
            df['quality'] = y
            
            # Save to CSV
            csv_path = input_dir / 'wine_quality_synthetic.csv'
            df.to_csv(csv_path, index=False)
            
            print(f"[INFO] Synthetic dataset saved to: {csv_path}")
        except Exception as e:
            print(f"[WARNING] Could not save synthetic data to CSV: {e}")
    
    def _split_data(self, X: np.ndarray, y: np.ndarray):
        """Split data into train/val/test and normalize."""
        # First split: 80% train+val, 20% test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if self.task == "classification" else None
        )
        
        # Second split: 75% train, 25% val (of the 80%)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, 
            stratify=y_temp if self.task == "classification" else None
        )
        
        # Normalize features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Convert to float32 for PyTorch
        self.X_train = self.X_train.astype(np.float32)
        self.X_val = self.X_val.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
        
        if self.task == "regression":
            self.y_train = self.y_train.astype(np.float32)
            self.y_val = self.y_val.astype(np.float32)
            self.y_test = self.y_test.astype(np.float32)
    
    def get_data_splits(self) -> Dict[str, np.ndarray]:
        """Return all data splits."""
        return {
            'X_train': self.X_train,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test
        }
    
    def get_info(self) -> Dict:
        """Return dataset metadata."""
        return {
            **self.dataset_info,
            'feature_names': self.feature_names,
            'train_size': len(self.X_train) if self.X_train is not None else 0,
            'val_size': len(self.X_val) if self.X_val is not None else 0,
            'test_size': len(self.X_test) if self.X_test is not None else 0
        }
    
    def get_sample_for_visualization(self) -> Tuple[np.ndarray, int]:
        """Get a random test sample for neuron activation visualization."""
        if self.X_test is not None and len(self.X_test) > 0:
            idx = np.random.randint(0, len(self.X_test))
            return self.X_test[idx], self.y_test[idx]
        return None, None
