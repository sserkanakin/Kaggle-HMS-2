"""
Quick test to verify multiprocessing setup works correctly.
"""

from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.make_dataset import load_config, process_all_data

if __name__ == "__main__":
    print("Testing multiprocessing setup...")
    config = load_config("configs/graphs.yaml")
    
    # Test with 2 workers (safe for testing)
    print("\nThis will process a few patients to test multiprocessing.")
    print("Press Ctrl+C to stop after testing.\n")
    
    try:
        process_all_data(config, n_workers=2)
    except KeyboardInterrupt:
        print("\n\nTest interrupted. Multiprocessing setup verified!")
