"""Test script to verify HMSDataset and HMSDataModule functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import HMSDataModule


def test_datamodule():
    """Test the HMSDataModule."""
    print("\n" + "="*60)
    print("Testing HMSDataModule")
    print("="*60)
    
    # Initialize datamodule
    datamodule = HMSDataModule(
        data_dir="data/processed",
        batch_size=4,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        num_workers=0,  # 0 for testing
        shuffle_seed=42,
    )
    
    # Setup
    print("\nSetting up datasets...")
    datamodule.setup(stage="fit")
    
    # Get dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    print(f"\nDataLoader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # Test one batch
    print(f"\nTesting one training batch...")
    batch = next(iter(train_loader))
    
    print(f"\nBatch structure:")
    print(f"  EEG graphs:  {len(batch['eeg_graphs'])} timesteps")
    print(f"    - First timestep batch: {batch['eeg_graphs'][0]}")
    print(f"    - Nodes per graph: {batch['eeg_graphs'][0].num_nodes // 4}")  # Divide by batch size
    print(f"  Spec graphs: {len(batch['spec_graphs'])} timesteps")
    print(f"    - First timestep batch: {batch['spec_graphs'][0]}")
    print(f"    - Nodes per graph: {batch['spec_graphs'][0].num_nodes // 4}")  # Divide by batch size
    print(f"  Targets shape: {batch['targets'].shape}")
    print(f"  Targets: {batch['targets']}")
    print(f"  Patient IDs: {batch['patient_ids']}")
    print(f"  Label IDs: {batch['label_ids']}")
    
    # Test class weights
    class_weights = datamodule.get_class_weights()
    print(f"\nClass weights: {class_weights}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_datamodule()
