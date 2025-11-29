# VinoGen-CyberCore 🧬🍷

<div align="center">

```
██╗   ██╗██╗███╗   ██╗ ██████╗  ██████╗ ███████╗███╗   ██╗
██║   ██║██║████╗  ██║██╔═══██╗██╔════╝ ██╔════╝████╗  ██║
██║   ██║██║██╔██╗ ██║██║   ██║██║  ███╗█████╗  ██╔██╗ ██║
╚██╗ ██╔╝██║██║╚██╗██║██║   ██║██║   ██║██╔══╝  ██║╚██╗██║
 ╚████╔╝ ██║██║ ╚████║╚██████╔╝╚██████╔╝███████╗██║ ╚████║
  ╚═══╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═══╝
                  NEUROgen SYSTEM
```

**A Hybrid Neural Network + Genetic Algorithm System for Wine Quality Prediction**

![Python Version](https://img.shields.io/badge/python-3.12.7-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production-brightgreen)
![Phase](https://img.shields.io/badge/phase-3%20complete-purple)

</div>

---

## Version 1.9 Updates (December 2025)

**Major Enhancements:**
-  **Interactive Menu System** - Matrix-style cyberpunk menu with 5 options
-  **Model Persistence** - Save/load trained models with full metadata
-  **Inference Mode** - Test predictions on random samples with confidence scores
-  **Advanced Visualizations** - NetworkX animations, probability heatmaps, regression plots
-  **Robust Error Handling** - Graceful degradation, no crashes on visualization failures
-  **Markdown Reports** - Professional mission reports with insights
-  **Data Persistence** - Synthetic datasets saved to CSV
-  **Directory Management** - Auto-creation of required directories
-  **Code Refactoring** - Modular structure with `__init__.py` files for cleaner imports

---

## Overview

**VinoGen-CyberCore** is a cutting-edge machine learning system that combines:

- 🧬 **Genetic Algorithms** for Neural Architecture Search (NAS)
- 🔮 **Dynamic MLPs** built with PyTorch
- 🎨 **Cyberpunk Terminal UI** using Rich library
- 📊 **Advanced Visualizations** with Matplotlib, Seaborn, and Plotly
- 🍷 **Wine Quality Classification & Regression**

The system evolves optimal neural network architectures through natural selection, creating high-performance models with minimal human intervention.

---

## Features

### Core Capabilities

 **Interactive Menu System** (NEW in Phase 3)
- Matrix-style cyberpunk menu
- 5 options: NEW RUN, LOAD CORE, INFERENCE, VIEW MODELS, EXIT
- Graceful error handling and keyboard interrupt support

 **Model Persistence** (NEW in Phase 3)
- Save trained models with metadata (timestamp, fitness, accuracy)
- Load models for inference
- Browse all saved models

 **Inference Mode** (NEW in Phase 3)
- Test loaded models on random samples
- Confidence scores with color-coded results (✓/✗)
- Sample accuracy calculation

 **Advanced Visualizations** (ENHANCED in Phase 3)
- NetworkX-based animated network flow (NEW)
- Probability heatmaps showing confidence (NEW)
- Regression analysis plots (NEW)
- All methods wrapped with error handling

 **Professional Reports** (NEW in Phase 3)
- Markdown mission reports with sections
- Executive summary, architecture, metrics, insights
- Links to all generated assets

 **Neuroevolution Engine**
- Evolves network topology (layers, neurons, activations)
- Genetic operators: Selection, Crossover, Mutation
- Fitness-based optimization

 **Dynamic Neural Networks**
- Variable architecture support
- Multiple activation functions (ReLU, Sigmoid, Tanh, LeakyReLU, ELU)
- Automatic training and validation

 **Educational & Verbose**
- Mathematical explanations (LaTeX equations in terminal)
- Real-time progress tracking
- Detailed logging and metrics

 **Stunning Visualizations**
- Network topology graphs
- Neuron activation flow animations (GIF)
- Learning curves and loss landscapes
- Confusion matrices and classification reports
- Interactive 3D plots with Plotly

 **Cyberpunk Terminal UI**
- Matrix-style boot sequences
- Neon color scheme (green, blue, purple, pink)
- Real-time dashboards and progress bars
- ASCII art headers

---

##  Project Structure

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
│   ├── models/                  # NEW: Saved models
│   │   ├── 20251128_163000_best_genome.pkl
│   │   └── ...
│   ├── network_topology.png    # Network architecture graph
│   ├── network_activation.gif  # NEW: Advanced NetworkX animation
│   ├── activation_flow.gif     # Original neuron activation animation
│   ├── learning_curves.png     # Training/validation curves
│   ├── confusion_matrix.png    # Classification performance
│   ├── probability_heatmap.png # NEW: Confidence visualization
│   ├── regression_analysis.png # NEW: Regression plots
│   ├── evolution_history.png   # GA evolution plot
│   ├── loss_landscape_3d.html  # Interactive 3D plot
│   ├── final_report.txt        # Comprehensive report
│   ├── MISSION_REPORT.md       # NEW: Markdown report
│   └── results.json            # Detailed results
│
├── input/                       # NEW: Input data
│   └── wine_quality_synthetic.csv # Generated dataset
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

##  Installation

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

##  Usage

### Quick Start (Phase 3 - Menu-Driven Interface)

```bash
python main.py
```

**You'll see the interactive menu:**

```
╔════════════════════════════════════════════════════╗
║         🧬 VINOGEN-CYBERCORE SYSTEM 🧬            ║
║            Neural Evolution Protocol               ║
╚════════════════════════════════════════════════════╝

[1]  NEW RUN       - Evolve new architecture
[2]  LOAD CORE     - Load saved model
[3]  INFERENCE     - Test model predictions
[4]  VIEW MODELS   - List saved models
[5]  EXIT          - Shutdown system

Select option:
```

### Menu Options

**Option 1: NEW RUN** - Train a new model
1. Boot cyberpunk interface
2. Load/generate wine quality dataset
3. Evolve neural architectures (10 generations, 20 individuals)
4. Train the best model
5. Evaluate on test set
6. Generate all visualizations (including new NetworkX animation, heatmaps)
7. **Save model to `output/models/`** (NEW)
8. **Generate markdown report** (NEW)
9. Display results dashboard
10. Return to menu

**Option 2: LOAD CORE** - Load a saved model
- Lists all models in `output/models/`
- Shows timestamp, fitness, accuracy
- Select by number
- Model ready for inference

**Option 3: INFERENCE** - Test predictions
- Requires loaded model (Option 1 or 2)
- Shows scanning animation
- Tests 5 random samples
- Displays results with confidence scores
- Color-coded: ✓ green (correct), ✗ red (error)

**Option 4: VIEW MODELS** - Browse saved models
- Non-interactive view of all saved models
- Shows metadata table

**Option 5: EXIT** - Graceful shutdown

📖 **For detailed usage guide, see [PHASE3_USAGE_GUIDE.md](PHASE3_USAGE_GUIDE.md)**

### Legacy Mode (Direct Run - Deprecated)

```bash
# This still works but bypasses the menu
# python main.py
# (Menu will appear - select Option 1)
```

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

##  How It Works

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

##  Output Files

After execution, check the `output/` directory:

### Visualizations

1. **network_topology.png** - Network architecture diagram
2. **network_activation.gif** - **NEW:** Advanced NetworkX animation with neural pulses
3. **activation_flow.gif** - Original animated neuron activations
4. **learning_curves.png** - Training/validation metrics
5. **confusion_matrix.png** - Classification performance (classification only)
6. **probability_heatmap.png** - **NEW:** Confidence visualization for sample predictions
7. **regression_analysis.png** - **NEW:** Predicted vs Actual with residuals (regression only)
8. **evolution_history.png** - GA fitness evolution
9. **loss_landscape_3d.html** - Interactive 3D loss surface

### Reports

- **MISSION_REPORT.md** - **NEW:** Professional markdown report with sections
- **final_report.txt** - Comprehensive text report
- **results.json** - Detailed results (JSON format)

### Saved Models (NEW)

- **output/models/YYYYMMDD_HHMMSS_best_genome.pkl** - Trained models with metadata

### Input Data (NEW)

- **input/wine_quality_synthetic.csv** - Generated synthetic dataset (5000 samples)

---

##  Screenshots

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
║           GENERATION 5 SUMMARY                           ║
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

##  Testing

Run unit tests:

```bash
pytest tests/
```

---

##  Mathematical Background

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

##  Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License.

---

##  Author

- **"Cerebros.cpp"** - Lead Developers 

**Neuroevolution AI Lab**
- Project: VinoGen-CyberCore
- Version: 1.0.0
- Year: 2025

---

##  Acknowledgments

- PyTorch team for the neural network framework
- Scikit-learn for preprocessing utilities
- Rich library for terminal UI
- Wine Quality dataset contributors

---

##  Support

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
