# VinoGen-CyberCore 🧬🍷⚡

<div align="center">

```
██╗   ██╗██╗███╗   ██╗ ██████╗  ██████╗ ███████╗███╗   ██╗
██║   ██║██║████╗  ██║██╔═══██╗██╔════╝ ██╔════╝████╗  ██║
██║   ██║██║██╔██╗ ██║██║   ██║██║  ███╗█████╗  ██╔██╗ ██║
╚██╗ ██╔╝██║██║╚██╗██║██║   ██║██║   ██║██╔══╝  ██║╚██╗██║
 ╚████╔╝ ██║██║ ╚████║╚██████╔╝╚██████╔╝███████╗██║ ╚████║
  ╚═══╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═══╝
                  CYBERCORE NEUROEVOLUTION SYSTEM
```

**A Hybrid Neural Network + Genetic Algorithm System for Wine Quality Prediction**

![Python Version](https://img.shields.io/badge/python-3.12.7-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production-brightgreen)

</div>

---

## 🌟 Overview

**VinoGen-CyberCore** is a cutting-edge machine learning system that combines:

- 🧬 **Genetic Algorithms** for Neural Architecture Search (NAS)
- 🔮 **Dynamic MLPs** built with PyTorch
- 🎨 **Cyberpunk Terminal UI** using Rich library
- 📊 **Advanced Visualizations** with Matplotlib, Seaborn, and Plotly
- 🍷 **Wine Quality Classification & Regression**

The system evolves optimal neural network architectures through natural selection, creating high-performance models with minimal human intervention.

---

## 🚀 Features

### Core Capabilities

✅ **Neuroevolution Engine**
- Evolves network topology (layers, neurons, activations)
- Genetic operators: Selection, Crossover, Mutation
- Fitness-based optimization

✅ **Dynamic Neural Networks**
- Variable architecture support
- Multiple activation functions (ReLU, Sigmoid, Tanh, LeakyReLU, ELU)
- Automatic training and validation

✅ **Educational & Verbose**
- Mathematical explanations (LaTeX equations in terminal)
- Real-time progress tracking
- Detailed logging and metrics

✅ **Stunning Visualizations**
- Network topology graphs
- Neuron activation flow animations (GIF)
- Learning curves and loss landscapes
- Confusion matrices and classification reports
- Interactive 3D plots with Plotly

✅ **Cyberpunk Terminal UI**
- Matrix-style boot sequences
- Neon color scheme (green, blue, purple, pink)
- Real-time dashboards and progress bars
- ASCII art headers

---

## 📁 Project Structure

```
VinoGen-CyberCore/
│
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── ARCHITECTURE.md              # System architecture documentation
│
├── config/                      # Configuration files
│   └── config.yaml             # System configuration
│
├── data/                        # Dataset directory
│   └── winequality.csv         # Wine dataset (optional)
│
├── output/                      # Generated outputs
│   ├── network_topology.png    # Network architecture graph
│   ├── activation_flow.gif     # Neuron activation animation
│   ├── learning_curves.png     # Training/validation curves
│   ├── confusion_matrix.png    # Classification performance
│   ├── evolution_history.png   # GA evolution plot
│   ├── loss_landscape_3d.html  # Interactive 3D plot
│   ├── final_report.txt        # Comprehensive report
│   └── results.json            # Detailed results
│
├── src/                         # Source code
│   ├── data/                   # Data handling
│   │   ├── __init__.py
│   │   └── data_handler.py     # Dataset loader & preprocessor
│   │
│   ├── models/                 # Neural network models
│   │   ├── __init__.py
│   │   └── mlp_model.py        # Dynamic MLP implementation
│   │
│   ├── genetic/                # Genetic algorithm
│   │   ├── __init__.py
│   │   └── genetic_optimizer.py # GA engine
│   │
│   ├── visualization/          # Visualization engine
│   │   ├── __init__.py
│   │   └── visualizer.py       # Plot generation
│   │
│   ├── ui/                     # User interface
│   │   ├── __init__.py
│   │   └── cyberpunk_ui.py     # Terminal UI
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       └── helpers.py          # Helper functions
│
├── tests/                       # Unit tests
├── assets/                      # Assets (logos, etc.)
├── docs/                        # Additional documentation
├── input/                       # Input data
└── scripts/                     # Utility scripts
```

---

## 🛠️ Installation

### Prerequisites

- **Python 3.12.7** (recommended)
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone Repository

```bash
cd Proyecto_Final
```

### Step 2: Create Virtual Environment

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with `pygraphviz`, you may need to install Graphviz separately:

- **Windows:** Download from [graphviz.org](https://graphviz.org/download/)
- **Linux:** `sudo apt-get install graphviz libgraphviz-dev`
- **Mac:** `brew install graphviz`

---

## 🎮 Usage

### Quick Start

```bash
python main.py
```

This will:
1. Boot the cyberpunk interface
2. Load/generate wine quality dataset
3. Evolve neural architectures (10 generations, 20 individuals)
4. Train the best model
5. Evaluate on test set
6. Generate all visualizations
7. Display results dashboard

### Configuration

Edit `config/config.yaml` or modify `src/utils/config.py` to customize:

```yaml
genetic_algorithm:
  population_size: 20        # Number of networks per generation
  generations: 10            # Evolution cycles
  mutation_rate: 0.3         # Mutation probability
  crossover_rate: 0.7        # Crossover probability

architecture:
  max_layers: 5              # Maximum hidden layers
  max_neurons: 256           # Maximum neurons per layer
  min_neurons: 16            # Minimum neurons per layer

training:
  epochs_per_genome: 50      # Training epochs per individual
  batch_size: 32             # Mini-batch size
```

### Custom Dataset

Place your wine quality CSV file in `data/winequality.csv` with format:

```csv
fixed_acidity,volatile_acidity,citric_acid,...,quality
7.4,0.7,0.0,...,5
```

If no file is found, the system generates synthetic data automatically.

---

## 🧬 How It Works

### 1. Genetic Algorithm

The system uses a genetic algorithm to search for optimal neural network architectures:

```
┌─────────────────────────────────────────┐
│  1. Initialize Population               │
│     - Random architectures              │
│     - Various layer configurations      │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  2. Evaluate Fitness                    │
│     - Train each network                │
│     - Measure validation performance    │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  3. Selection                           │
│     - Tournament selection              │
│     - Keep elite performers             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  4. Crossover                           │
│     - Combine parent architectures      │
│     - Create offspring                  │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  5. Mutation                            │
│     - Add/remove neurons                │
│     - Change activation functions       │
│     - Adjust learning rates             │
└─────────────────────────────────────────┘
              ↓
         Repeat for N generations
```

### 2. Neural Network

Dynamic MLP with evolved architecture:

**Forward Pass:**
```
z[l] = W[l] @ a[l-1] + b[l]
a[l] = σ(z[l])
```

**Loss Functions:**
- **Classification:** Cross-Entropy Loss
  ```
  L = -Σ y_true * log(y_pred)
  ```
- **Regression:** Mean Squared Error
  ```
  MSE = (1/n) * Σ(y_true - y_pred)²
  ```

**Backpropagation:**
```
∂L/∂W[l] = ∂L/∂a[l] * ∂a[l]/∂z[l] * ∂z[l]/∂W[l]
```

**Optimization:**
```
W ← W - α * ∂L/∂W
```

### 3. Fitness Function

Balances accuracy and complexity:

```python
fitness = accuracy  # For classification
fitness = R²        # For regression
```

Elite genomes are preserved across generations.

---

## 📊 Output Files

After execution, check the `output/` directory:

### Visualizations

1. **network_topology.png** - Network architecture diagram
2. **activation_flow.gif** - Animated neuron activations
3. **learning_curves.png** - Training/validation metrics
4. **confusion_matrix.png** - Classification performance (classification only)
5. **evolution_history.png** - GA fitness evolution
6. **loss_landscape_3d.html** - Interactive 3D loss surface

### Reports

- **final_report.txt** - Comprehensive text report
- **results.json** - Detailed results (JSON format)

---

## 🎨 Screenshots

### Terminal Boot Sequence
```
██╗   ██╗██╗███╗   ██╗ ██████╗  ██████╗ ███████╗███╗   ██╗
[SYSTEM] Initializing Neural Core...
[QUANTUM] Loading Genetic Algorithm Engine...
[MATRIX] Establishing Data Pipeline...
```

### Evolution Progress
```
╔══════════════════════════════════════════════════════════╗
║           GENERATION 5 SUMMARY                          ║
╚══════════════════════════════════════════════════════════╝

🏆 TOP 5 ARCHITECTURES:
──────────────────────────────────────────────────────────
  Rank 1:
    Architecture: 128 → 64 → 32
    Activations:  relu, tanh, sigmoid
    Learning Rate: 0.002134
    Fitness:      0.956234
```

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

---

## 📚 Mathematical Background

### Activation Functions

- **ReLU:** `f(x) = max(0, x)`
- **Sigmoid:** `f(x) = 1 / (1 + e^(-x))`
- **Tanh:** `f(x) = tanh(x)`
- **Leaky ReLU:** `f(x) = max(0.01x, x)`

### Genetic Operators

**Tournament Selection:**
- Select k random individuals
- Choose the best among them

**Single-Point Crossover:**
- Pick crossover point
- Swap layer configurations

**Mutation:**
- Add/remove neurons
- Add/remove layers
- Change activation functions

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Neuroevolution AI Lab**
- Project: VinoGen-CyberCore
- Version: 1.0.0
- Year: 2025

---

## 🙏 Acknowledgments

- PyTorch team for the neural network framework
- Scikit-learn for preprocessing utilities
- Rich library for terminal UI
- Wine Quality dataset contributors

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the documentation in `docs/`

---

<div align="center">

**⚡ Built with Python, PyTorch, and Cyberpunk Aesthetics ⚡**

```
     ╔═══════════════════════════════════════╗
     ║  EVOLUTION COMPLETE. SYSTEM READY.    ║
     ╚═══════════════════════════════════════╝
```

</div>
