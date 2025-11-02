#!/usr/bin/env python3
"""
Quick pre-flight check before running preprocessing
Verifies all data files and configurations are in place
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_file_exists(path, name):
    """Check if file exists and print status"""
    if os.path.exists(path):
        size = os.path.getsize(path) if os.path.isfile(path) else "dir"
        print(f"  ✓ {name}: {path}")
        return True
    else:
        print(f"  ✗ {name} MISSING: {path}")
        return False

def check_directory(path, name, min_files=0):
    """Check if directory exists and has files"""
    if not os.path.exists(path):
        print(f"  ✗ {name} directory MISSING: {path}")
        return False
    
    if os.path.isdir(path):
        files = list(Path(path).glob("*"))
        if len(files) >= min_files:
            print(f"  ✓ {name}: {path} ({len(files)} files)")
            return True
        else:
            print(f"  ⚠ {name}: {path} (only {len(files)} files, expected >= {min_files})")
            return False
    return False

print("=" * 70)
print("HMS Graph Preprocessing - Pre-flight Check")
print("=" * 70)

all_ok = True

print("\n[1/4] Checking configuration files...")
all_ok &= check_file_exists("configs/graphs.yaml", "Graph config")

print("\n[2/4] Checking input data...")
all_ok &= check_file_exists("data/raw/train.csv", "Training CSV")
all_ok &= check_directory("data/raw/train_eegs", "EEG files", min_files=100)
all_ok &= check_directory("data/raw/train_spectrograms", "Spectrogram files", min_files=100)

print("\n[3/4] Checking output directory...")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"  ✓ Output directory ready: {output_dir}")

print("\n[4/4] Checking Python environment...")
try:
    from omegaconf import OmegaConf
    import pandas as pd
    import torch
    import torch_geometric
    print("  ✓ All required packages installed")
except ImportError as e:
    print(f"  ✗ Missing package: {e}")
    all_ok = False

print("\n" + "=" * 70)
if all_ok:
    print("✅ ALL CHECKS PASSED - Ready to run!")
    print("\nTo start preprocessing:")
    print("  ./run_preprocessing.sh")
    print("\nOr manually:")
    print("  python3 src/data/make_graph_dataset.py --workers 8")
else:
    print("❌ SOME CHECKS FAILED - Please fix issues above before running")
    sys.exit(1)

print("=" * 70)
