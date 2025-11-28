# Installation Guide

## Quick Install (Recommended)

### Step 1: Setup Virtual Environment

```powershell
# Navigate to project directory
cd Proyecto_Final

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

That's it! The system is ready to run.

## Troubleshooting

### Common Issues

#### 1. pygraphviz Installation Error (Safe to Ignore)

**Error Message:**
```
ERROR: Failed building wheel for pygraphviz
```

**Solution:**
This package is **optional** and **not required** for the system to work. The visualization system uses `networkx` instead, which works perfectly on all platforms.

If you still want to install pygraphviz (advanced users only):

**Windows:**
1. Download Graphviz installer from: https://graphviz.org/download/
2. Install to: `C:\Program Files\Graphviz`
3. Add to PATH: `C:\Program Files\Graphviz\bin`
4. Restart terminal
5. Install pygraphviz: `pip install pygraphviz`

**Alternative:** The system works perfectly without it!

#### 2. PyTorch Installation Issues

**Error:** Slow download or CUDA version mismatch

**Solution:**
```powershell
# CPU-only version (smaller, faster download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU version (if you have NVIDIA GPU):**
```powershell
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Kaleido Installation Issues

**Error:** Plotly static image export fails

**Solution:**
```powershell
pip install kaleido --force-reinstall
```

**Alternative:** The system will still generate interactive HTML plots.

#### 4. Microsoft Visual C++ Build Tools Required

**Error:** `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution:**
Install Visual C++ Build Tools:
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++" workload
3. Restart terminal and retry installation

**Alternative:** Most packages have pre-built wheels and don't require compilation.

#### 5. Permission Errors

**Error:** `[WinError 5] Access is denied`

**Solution:**
Run PowerShell as Administrator or use:
```powershell
pip install -r requirements.txt --user
```

### Minimal Installation (If Issues Persist)

If you encounter multiple installation issues, use this minimal requirements file:

Create `requirements-minimal.txt`:
```
numpy>=1.24.0
pandas>=2.0.0
torch>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.1
imageio>=2.31.0
rich>=13.3.0
colorama>=0.4.6
tqdm>=4.65.0
pyyaml>=6.0
pytest>=7.4.0
```

Install:
```powershell
pip install -r requirements-minimal.txt
```

**Note:** This removes optional packages (plotly 3D plots, advanced formatting tools) but maintains core functionality.

## Platform-Specific Instructions

### Windows 10/11

**Prerequisites:**
- Python 3.12.7 (download from python.org)
- PowerShell 5.1+ (built-in)

**Installation:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**If Execution Policy Error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux (Ubuntu/Debian)

**Prerequisites:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

**Installation:**
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### macOS

**Prerequisites:**
```bash
brew install python@3.12
```

**Installation:**
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Apple Silicon (M1/M2) Note:**
PyTorch has native Apple Silicon support:
```bash
pip install torch torchvision
```

## Verification

### Test Installation

```powershell
python -c "import torch; import pandas; import rich; print('✅ All core packages installed!')"
```

### Run Tests

```powershell
pytest tests/ -v
```

Expected output: All tests pass ✅

### Quick Run

```powershell
python main.py
```

You should see the cyberpunk boot sequence!

## Dependency Overview

### Essential (Required)
- **numpy, pandas, scipy:** Data processing
- **torch, torchvision:** Neural networks
- **scikit-learn:** ML utilities
- **matplotlib, seaborn:** Basic plotting
- **networkx:** Graph visualization
- **rich, colorama, tqdm:** Terminal UI
- **pyyaml:** Configuration

### Optional (Enhanced Features)
- **plotly, kaleido:** Interactive 3D visualizations
- **pygraphviz:** Advanced graph layouts (requires Graphviz)
- **imageio-ffmpeg:** Video generation (animated plots)
- **pyfiglet, tabulate:** Aesthetic enhancements
- **pytest, black, flake8, mypy:** Development tools

### Development Only
- **pytest, pytest-cov:** Testing
- **black:** Code formatting
- **flake8:** Linting
- **mypy:** Type checking

## Virtual Environment Management

### Activate Environment

**Windows:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### Deactivate Environment

```powershell
deactivate
```

### Delete and Recreate Environment

```powershell
# Deactivate first
deactivate

# Delete
Remove-Item -Recurse -Force venv

# Recreate
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Updating Dependencies

```powershell
# Update pip first
python -m pip install --upgrade pip

# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade torch
```

## Offline Installation

If you need to install on a machine without internet:

### 1. Download packages on connected machine

```powershell
pip download -r requirements.txt -d packages/
```

### 2. Transfer `packages/` folder to offline machine

### 3. Install from local packages

```powershell
pip install --no-index --find-links packages/ -r requirements.txt
```

## Docker Installation (Alternative)

If you prefer containerized deployment:

Create `Dockerfile`:
```dockerfile
FROM python:3.12.7-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

Build and run:
```powershell
docker build -t vinogen-cybercore .
docker run -v ${PWD}/output:/app/output vinogen-cybercore
```

## Performance Optimization

### Use conda (Alternative to venv)

```powershell
conda create -n vinogen python=3.12.7
conda activate vinogen
pip install -r requirements.txt
```

**Benefits:**
- Better binary package management
- Pre-compiled scientific libraries
- Faster installation

### Pre-compiled Wheels

Download pre-compiled wheels from: https://www.lfd.uci.edu/~gohlke/pythonlibs/

For packages like:
- numpy-mkl (faster NumPy)
- scipy
- scikit-learn

## Getting Help

### Check Python Version

```powershell
python --version
# Should output: Python 3.12.7
```

### Check Installed Packages

```powershell
pip list
```

### Check Package Information

```powershell
pip show torch
```

### Generate Requirements from Environment

```powershell
pip freeze > requirements-frozen.txt
```

## Next Steps

After successful installation:

1. ✅ Run tests: `pytest tests/`
2. ✅ Quick start: `python main.py`
3. ✅ Check output: `ls output/`
4. ✅ Read docs: `docs/QUICKSTART.md`

---

**Installation Complete! 🎉**

If you encounter any issues not covered here, check the main README.md or QUICKSTART.md for additional guidance.
