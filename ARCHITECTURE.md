# VinoGen-CyberCore: System Architecture

## System Overview

VinoGen-CyberCore is a neuroevolution system that combines genetic algorithms with neural networks to solve wine quality prediction problems. The architecture follows a modular design pattern with clear separation of concerns.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE LAYER                        │
│                         (Cyberpunk Terminal UI)                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Boot Sequence │ Progress Bars │ Tables │ Dashboards        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                             │
│                         (Main Controller)                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Pipeline Management │ Workflow Control │ State Management  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ↓                           ↓
┌──────────────────────────────┐  ┌──────────────────────────────┐
│   DATA PROCESSING LAYER      │  │   EVOLUTION ENGINE LAYER     │
│  ┌────────────────────────┐  │  │  ┌────────────────────────┐  │
│  │ Data Loading           │  │  │  │ Population Init        │  │
│  │ Preprocessing          │  │  │  │ Fitness Evaluation     │  │
│  │ Train/Val/Test Split   │  │  │  │ Selection              │  │
│  │ Feature Scaling        │  │  │  │ Crossover              │  │
│  │ Synthetic Generation   │  │  │  │ Mutation               │  │
│  └────────────────────────┘  │  │  └────────────────────────┘  │
└──────────────────────────────┘  └──────────────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     NEURAL NETWORK LAYER                             │
│                        (Dynamic MLP)                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Variable Architecture │ Forward Pass │ Backpropagation     │   │
│  │  Multiple Activations  │ Loss Functions │ Optimization      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     VISUALIZATION LAYER                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Topology Graphs │ Animations │ Learning Curves            │   │
│  │  3D Landscapes   │ Confusion Matrices │ Reports            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        STORAGE LAYER                                 │
│         (Output Files: Images, GIFs, HTML, JSON, TXT)               │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Data Processing Module (`src/data/`)

**Purpose:** Handle all data-related operations

**Components:**
- `DataHandler`: Main data management class
  - Load CSV datasets
  - Generate synthetic wine data
  - Preprocess and normalize features
  - Split into train/validation/test sets

**Key Features:**
- Automatic synthetic data generation if real data unavailable
- StandardScaler for feature normalization
- Stratified splitting for classification
- Support for both classification and regression tasks

**Dependencies:**
- pandas, numpy, scikit-learn

---

### 2. Neural Network Module (`src/models/`)

**Purpose:** Implement dynamic neural networks

**Components:**
- `DynamicMLP`: PyTorch-based multilayer perceptron
  - Variable number of hidden layers
  - Configurable neurons per layer
  - Multiple activation functions
  - Automatic loss function selection
  - Training and evaluation methods

**Architecture:**
```python
Input → [Hidden Layer 1] → [Hidden Layer 2] → ... → Output
         (activation)        (activation)
```

**Key Features:**
- Genome-driven architecture
- Adam optimizer
- Mini-batch training
- Validation during training
- Activation storage for visualization

**Dependencies:**
- torch, numpy

---

### 3. Genetic Algorithm Module (`src/genetic/`)

**Purpose:** Evolve optimal network architectures

**Components:**
- `GeneticOptimizer`: Main GA engine
  - Population initialization
  - Fitness evaluation
  - Selection (tournament)
  - Crossover (single-point)
  - Mutation (multi-type)
  
- `Genome`: Network architecture representation
  - Hidden layers specification
  - Activation functions
  - Learning rate
  - Fitness score

**Evolution Process:**
1. Initialize random population
2. Evaluate fitness (train & validate each network)
3. Select best performers (tournament selection)
4. Create offspring (crossover)
5. Apply mutations
6. Repeat for N generations

**Mutation Types:**
- Neuron count adjustment
- Layer addition/removal
- Activation function change
- Learning rate adjustment

**Dependencies:**
- numpy, copy, random

---

### 4. Visualization Module (`src/visualization/`)

**Purpose:** Generate visual outputs

**Components:**
- `Visualizer`: Comprehensive visualization engine
  - Network topology graphs (NetworkX)
  - Activation flow animations (ImageIO)
  - Learning curves (Matplotlib)
  - Confusion matrices (Seaborn)
  - 3D loss landscapes (Plotly)
  - Evolution history plots

**Output Formats:**
- PNG images (high resolution)
- GIF animations
- Interactive HTML (Plotly)
- Text reports

**Dependencies:**
- matplotlib, seaborn, plotly, networkx, imageio, PIL

---

### 5. User Interface Module (`src/ui/`)

**Purpose:** Provide cyberpunk-themed terminal interface

**Components:**
- `CyberpunkUI`: Rich-based terminal UI
  - Boot sequences
  - Progress bars
  - Data tables
  - Status panels
  - Color-coded logging
  - ASCII art headers

**Color Scheme:**
- Neon Green: `#39FF14` (success, primary)
- Electric Blue: `#00FFFF` (info, secondary)
- Deep Purple: `#9D00FF` (system)
- Hot Pink: `#FF10F0` (highlights)
- Cyber Yellow: `#FFFF00` (warnings)

**Dependencies:**
- rich, pyfiglet, colorama

---

### 6. Utilities Module (`src/utils/`)

**Purpose:** Shared utilities and configuration

**Components:**
- `Config`: Central configuration management
  - Project settings
  - GA parameters
  - Training hyperparameters
  - Visualization options
  - YAML save/load
  
- `helpers.py`: Utility functions
  - Random seed setting
  - Timing decorator
  - JSON I/O
  - Early stopping
  - Device selection
  - Number formatting

**Dependencies:**
- pyyaml, numpy, torch

---

## Data Flow

### Training Pipeline

```
1. Data Loading
   ↓
2. Dataset Split (60% train, 20% val, 20% test)
   ↓
3. Feature Normalization
   ↓
4. Genetic Evolution Loop:
   For each generation:
     For each genome:
       - Create network
       - Train on training set
       - Evaluate on validation set
       - Store fitness
     - Selection
     - Crossover
     - Mutation
   ↓
5. Final Training (best genome)
   ↓
6. Test Set Evaluation
   ↓
7. Visualization Generation
   ↓
8. Results Export
```

### Genome Structure

```python
Genome = {
    'hidden_layers': [128, 64, 32],              # Neuron counts
    'activation_functions': ['relu', 'tanh', 'sigmoid'],
    'learning_rate': 0.002134,
    'fitness': 0.956234,
    'generation': 5
}
```

---

## Design Patterns

### 1. Modular Architecture
- Each module has single responsibility
- Clear interfaces between components
- Easy to extend and modify

### 2. Configuration-Driven
- Central configuration class
- YAML file support
- Runtime parameter adjustment

### 3. Factory Pattern
- Dynamic network creation from genomes
- Flexible architecture instantiation

### 4. Strategy Pattern
- Multiple activation functions
- Different loss functions
- Various mutation strategies

---

## Performance Considerations

### Memory Management
- Mini-batch training reduces memory footprint
- Gradient computation only when needed
- Activation storage optional

### Computational Efficiency
- PyTorch GPU acceleration support
- Vectorized operations with NumPy
- Early stopping prevents overtraining

### Parallelization Opportunities
- Population evaluation (future enhancement)
- Visualization generation
- Data preprocessing

---

## Extension Points

### Adding New Features

1. **New Activation Functions:**
   ```python
   # In mlp_model.py, add to _get_activation()
   'swish': lambda x: x * torch.sigmoid(x)
   ```

2. **Custom Mutation Operators:**
   ```python
   # In genetic_optimizer.py, add to mutate()
   if random.random() < self.mutation_rate:
       # Your custom mutation
   ```

3. **Additional Visualizations:**
   ```python
   # In visualizer.py, add new method
   def plot_custom_viz(self, data, filename):
       # Your visualization code
   ```

4. **New Datasets:**
   ```python
   # In data_handler.py, modify _process_real_data()
   # Add dataset-specific preprocessing
   ```

---

## Dependencies Graph

```
main.py
├── src.data.DataHandler
│   ├── pandas
│   ├── numpy
│   └── sklearn
│
├── src.models.DynamicMLP
│   ├── torch
│   └── numpy
│
├── src.genetic.GeneticOptimizer
│   ├── numpy
│   └── random
│
├── src.visualization.Visualizer
│   ├── matplotlib
│   ├── seaborn
│   ├── plotly
│   ├── networkx
│   ├── imageio
│   └── PIL
│
├── src.ui.CyberpunkUI
│   ├── rich
│   ├── pyfiglet
│   └── colorama
│
└── src.utils
    ├── pyyaml
    ├── torch
    └── numpy
```

---

## Error Handling

### Graceful Degradation
- Missing dataset → Generate synthetic data
- Visualization failure → Continue with warning
- GPU unavailable → Fall back to CPU

### User Feedback
- Color-coded error messages
- Detailed logging
- Progress indicators

---

## Testing Strategy

### Unit Tests
- Data preprocessing
- Network forward/backward pass
- Genetic operators
- Utility functions

### Integration Tests
- Full pipeline execution
- Visualization generation
- Configuration loading

### Performance Tests
- Training speed
- Memory usage
- Scalability

---

## Future Enhancements

1. **Multi-GPU Support:** Parallel genome evaluation
2. **Hyperparameter Optimization:** Automated tuning
3. **Model Export:** ONNX, TorchScript formats
4. **Real-time Monitoring:** Web dashboard
5. **Ensemble Methods:** Combine multiple evolved networks
6. **Transfer Learning:** Pre-trained feature extractors

---

## Conclusion

VinoGen-CyberCore demonstrates a production-grade implementation of neuroevolution, combining modern deep learning frameworks with evolutionary computation in an aesthetically pleasing package.
