"""
Quick verification script for Phase 3 implementation.
Tests all imports and basic functionality without running full training.
"""

import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

print("=" * 70)
print("PHASE 3 VERIFICATION SCRIPT")
print("=" * 70)

# Test 1: Core imports
print("\n[1/8] Testing core imports...")
try:
    from src.data import DataHandler
    from src.models import DynamicMLP
    from src.genetic import GeneticOptimizer, Genome
    from src.visualization import Visualizer
    from src.ui import CyberpunkUI
    print("✅ Core imports successful")
except Exception as e:
    print(f"❌ Core imports failed: {e}")
    sys.exit(1)

# Test 2: Utils imports
print("\n[2/8] Testing utils imports...")
try:
    from src.utils import (
        Config, MATH_EQUATIONS, set_random_seeds, 
        EarlyStopping, save_json, get_device,
        save_model, load_model, list_saved_models,
        create_markdown_report, ensure_directories
    )
    print("✅ Utils imports successful (including Phase 3 functions)")
except Exception as e:
    print(f"❌ Utils imports failed: {e}")
    sys.exit(1)

# Test 3: UI menu methods
print("\n[3/8] Testing UI menu methods...")
try:
    ui = CyberpunkUI()
    # Check methods exist
    assert hasattr(ui, 'show_main_menu'), "Missing show_main_menu"
    assert hasattr(ui, 'show_model_selection'), "Missing show_model_selection"
    assert hasattr(ui, 'show_inference_scanning'), "Missing show_inference_scanning"
    assert hasattr(ui, 'show_inference_results'), "Missing show_inference_results"
    assert hasattr(ui, 'show_loading_animation'), "Missing show_loading_animation"
    print("✅ All UI menu methods exist")
except Exception as e:
    print(f"❌ UI methods check failed: {e}")
    sys.exit(1)

# Test 4: Visualizer methods
print("\n[4/8] Testing visualizer methods...")
try:
    viz = Visualizer(output_dir="output")
    # Check new Phase 3 methods exist
    assert hasattr(viz, 'animate_network_flow'), "Missing animate_network_flow"
    assert hasattr(viz, 'plot_probability_heatmap'), "Missing plot_probability_heatmap"
    assert hasattr(viz, 'plot_regression_analysis'), "Missing plot_regression_analysis"
    print("✅ All Phase 3 visualization methods exist")
except Exception as e:
    print(f"❌ Visualizer methods check failed: {e}")
    sys.exit(1)

# Test 5: Directory creation
print("\n[5/8] Testing directory creation...")
try:
    ensure_directories(ROOT_DIR)
    
    # Check directories exist
    assert (ROOT_DIR / 'input').exists(), "input/ not created"
    assert (ROOT_DIR / 'output').exists(), "output/ not created"
    assert (ROOT_DIR / 'output' / 'models').exists(), "output/models/ not created"
    
    print("✅ All directories created successfully")
    print(f"   - input/")
    print(f"   - output/")
    print(f"   - output/models/")
except Exception as e:
    print(f"❌ Directory creation failed: {e}")
    sys.exit(1)

# Test 6: Config loading
print("\n[6/8] Testing configuration...")
try:
    config = Config()
    config.save_to_yaml()
    
    assert config.GA_POPULATION_SIZE > 0, "Invalid population size"
    assert config.GA_GENERATIONS > 0, "Invalid generations"
    assert (ROOT_DIR / 'config' / 'config.yaml').exists(), "config.yaml not created"
    
    print("✅ Configuration loaded and saved")
    print(f"   - Population: {config.GA_POPULATION_SIZE}")
    print(f"   - Generations: {config.GA_GENERATIONS}")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# Test 7: Main class initialization
print("\n[7/8] Testing main system initialization...")
try:
    from main import VinoGenCyberCore
    system = VinoGenCyberCore()
    
    # Check Phase 3 attributes
    assert hasattr(system, 'loaded_model'), "Missing loaded_model attribute"
    assert hasattr(system, 'loaded_genome'), "Missing loaded_genome attribute"
    assert hasattr(system, 'loaded_history'), "Missing loaded_history attribute"
    assert hasattr(system, 'loaded_metrics'), "Missing loaded_metrics attribute"
    
    # Check Phase 3 methods
    assert hasattr(system, 'load_core'), "Missing load_core method"
    assert hasattr(system, 'run_inference'), "Missing run_inference method"
    assert hasattr(system, 'view_models'), "Missing view_models method"
    
    print("✅ Main system initialized with Phase 3 features")
except Exception as e:
    print(f"❌ Main system initialization failed: {e}")
    sys.exit(1)

# Test 8: matplotlib backend
print("\n[8/8] Testing matplotlib backend...")
try:
    import matplotlib
    backend = matplotlib.get_backend()
    print(f"✅ matplotlib backend: {backend}")
    if backend.lower() == 'agg':
        print("   ✅ Agg backend active (terminal-safe)")
    else:
        print(f"   ⚠️  Backend is {backend}, should be 'Agg' for terminal safety")
except Exception as e:
    print(f"❌ matplotlib check failed: {e}")

# Final summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE!")
print("=" * 70)
print("\n✅ All Phase 3 components verified successfully!")
print("\nYou can now run the main system:")
print("  python main.py")
print("\nDocumentation:")
print("  - PHASE3_USAGE_GUIDE.md - User guide")
print("  - PHASE3_IMPLEMENTATION.md - Technical details")
print("  - PHASE3_COMPLETION.md - Summary")
print("\n" + "=" * 70)
