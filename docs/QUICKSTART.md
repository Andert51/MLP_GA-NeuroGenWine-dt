# VinoGen-CyberCore: Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Setup Environment

```powershell
# Navigate to project directory
cd Proyecto_Final

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the System

```powershell
python main.py
```

That's it! The system will:
1. Boot with cyberpunk animation
2. Generate synthetic wine dataset (if no real data found)
3. Evolve neural architectures for 10 generations
4. Train the best model
5. Generate visualizations
6. Display results

### Step 3: Check Outputs

All outputs are saved to `output/` directory:

```
output/
├── network_topology.png       # Your evolved network architecture
├── activation_flow.gif        # Neural activation animation
├── learning_curves.png        # Training progress
├── confusion_matrix.png       # Classification performance
├── evolution_history.png      # GA evolution plot
├── loss_landscape_3d.html     # Interactive 3D visualization
├── final_report.txt           # Comprehensive report
└── results.json               # Detailed results
```

## 📝 Configuration (Optional)

Edit `src/utils/config.py` to customize:

```python
# Quick tweaks for faster execution
GA_POPULATION_SIZE = 10      # Reduce from 20
GA_GENERATIONS = 5           # Reduce from 10
EPOCHS_PER_GENOME = 30       # Reduce from 50

# For better results (slower)
GA_POPULATION_SIZE = 30
GA_GENERATIONS = 20
EPOCHS_PER_GENOME = 100
```

## 🍷 Using Your Own Dataset

Place your CSV file at `data/winequality.csv`:

```csv
fixed_acidity,volatile_acidity,citric_acid,residual_sugar,...,quality
7.4,0.70,0.00,1.9,...,5
```

Required columns:
- Features: Any numeric columns
- Target: `quality`, `type`, `label`, or `class` column

## 🎨 UI Features

### Color Coding
- 🟢 **Green:** Success messages, best results
- 🔵 **Blue:** Information, system status
- 🟣 **Purple:** System operations
- 🔴 **Pink:** Highlights, special data
- 🟡 **Yellow:** Warnings, important notes

### Progress Tracking
- Real-time progress bars for all operations
- Genome evaluation status
- Training epoch updates
- Visualization generation progress

## 🧬 Understanding the Output

### Network Topology
Shows the evolved architecture:
- Input layer (wine features)
- Hidden layers (evolved structure)
- Output layer (quality prediction)
- Activation functions per layer

### Activation Flow Animation
Watch data propagate through neurons:
- Each frame = one layer's activation
- Color intensity = neuron activation strength
- Shows how the network processes information

### Learning Curves
Training performance over time:
- Training loss (should decrease)
- Validation loss (should decrease)
- Training accuracy (should increase)
- Validation accuracy (should increase)

### Evolution History
Genetic algorithm progress:
- Best fitness per generation
- Average population fitness
- Population diversity

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Ensure virtual environment is activated and dependencies installed
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "pygraphviz installation failed"
**Solution:** Install Graphviz separately
```powershell
# Windows: Download from https://graphviz.org/download/
# Then: pip install pygraphviz
```

### Issue: "CUDA out of memory"
**Solution:** System uses CPU by default. If you enabled GPU, reduce batch size:
```python
BATCH_SIZE = 16  # In config.py
```

### Issue: "Slow execution"
**Solution:** Reduce evolution parameters:
```python
GA_POPULATION_SIZE = 10
GA_GENERATIONS = 5
EPOCHS_PER_GENOME = 20
```

## 📊 Performance Tips

### Fast Execution (~5 minutes)
```python
GA_POPULATION_SIZE = 5
GA_GENERATIONS = 3
EPOCHS_PER_GENOME = 20
```

### Balanced (~15 minutes)
```python
GA_POPULATION_SIZE = 10
GA_GENERATIONS = 8
EPOCHS_PER_GENOME = 40
```

### Best Results (~45 minutes)
```python
GA_POPULATION_SIZE = 30
GA_GENERATIONS = 20
EPOCHS_PER_GENOME = 100
```

## 🎯 Expected Results

### Typical Performance
- **Classification Accuracy:** 70-85%
- **Regression R² Score:** 0.5-0.75
- **Best Fitness:** >0.80

### Architecture Examples
Common evolved structures:
- `128 → 64 → 32` (3 layers)
- `256 → 128 → 64` (3 layers)
- `64 → 32` (2 layers)

### Activation Patterns
Most successful combinations:
- ReLU → ReLU → Sigmoid
- ReLU → Tanh → Sigmoid
- LeakyReLU → ReLU → Tanh

## 🧪 Testing the System

Run unit tests:

```powershell
pytest tests/ -v
```

Expected output:
```
tests/test_main.py::TestDataHandler::test_synthetic_data_generation PASSED
tests/test_main.py::TestDataHandler::test_data_splits PASSED
tests/test_main.py::TestDynamicMLP::test_network_creation PASSED
...
```

## 📚 Next Steps

1. **Experiment with parameters:** Try different population sizes and mutation rates
2. **Try real datasets:** Add your own wine quality CSV
3. **Analyze visualizations:** Study the evolved architectures
4. **Compare runs:** Run multiple times and compare results
5. **Extend the system:** Add new activation functions or mutation operators

## 💡 Tips

- **Let it run:** Evolution takes time, patience pays off
- **Watch the UI:** The cyberpunk interface shows everything happening
- **Check outputs:** Visualizations reveal insights about the learning process
- **Experiment:** Modify config values and observe changes
- **Save results:** Each run creates unique output files

## 🎮 Advanced Usage

### Custom Task
Change from classification to regression:

```python
# In config.py
TASK = "regression"  # Instead of "classification"
```

### Multiple Runs
Compare different configurations:

```powershell
# Run 1: Small population
python main.py

# Rename output folder
mv output output_run1

# Run 2: Large population
# (modify config first)
python main.py
```

### Batch Experimentation
Create a script to run multiple configurations:

```python
# experiment.py
from src.utils import Config
from main import VinoGenCyberCore

configs = [
    {'population': 10, 'generations': 5},
    {'population': 20, 'generations': 10},
    {'population': 30, 'generations': 15}
]

for cfg in configs:
    config = Config()
    config.GA_POPULATION_SIZE = cfg['population']
    config.GA_GENERATIONS = cfg['generations']
    
    system = VinoGenCyberCore(config)
    system.run()
```

## 🏆 Success Indicators

Your system is working well if:
- ✅ Best fitness increases over generations
- ✅ Validation accuracy > 70%
- ✅ Learning curves show convergence
- ✅ Evolved architecture makes sense (gradual decrease in neurons)
- ✅ Test performance matches validation performance

## 📞 Need Help?

- Check `ARCHITECTURE.md` for system details
- Review `README.md` for comprehensive documentation
- Inspect `output/final_report.txt` for run-specific information
- Look at visualization files for insights

---

**Happy Evolving! 🧬⚡**
