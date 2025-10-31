"""Main training script for HMS Multi-Modal GNN."""

import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import HMSDataModule
from src.lightning_trainer import HMSLightningModule


def train(
    config_path: str = "configs/model.yaml",
    wandb_project: str = "hms-brain-activity",
    wandb_name: str | None = None,
    resume_from_checkpoint: str | None = None,
):
    """Train HMS Multi-Modal GNN model.
    
    Parameters
    ----------
    config_path : str
        Path to model configuration file
    wandb_project : str
        WandB project name
    wandb_name : str, optional
        WandB run name (auto-generated if None)
    resume_from_checkpoint : str, optional
        Path to checkpoint to resume from
    """
    # Load configuration
    config = OmegaConf.load(config_path)
    
    print("\n" + "="*60)
    print("HMS Multi-Modal GNN Training")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"WandB Project: {wandb_project}")
    print(f"WandB Run: {wandb_name or 'auto-generated'}")
    print("="*60 + "\n")
    
    # Initialize DataModule
    print("Initializing DataModule...")
    datamodule = HMSDataModule(
        data_dir=config.data.get('data_dir', 'data/processed'),
        batch_size=config.training.batch_size,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        shuffle_seed=config.data.shuffle_seed,
    )
    
    # Setup to get class weights
    datamodule.setup(stage="fit")
    class_weights = datamodule.get_class_weights() if config.training.use_class_weights else None
    
    # Initialize Lightning Module
    print("Initializing Model...")
    model = HMSLightningModule(
        model_config=config.model,
        num_classes=config.model.num_classes,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        class_weights=class_weights,
        scheduler_config=config.training.scheduler,
    )
    
    # Print model info
    model_info = model.get_model_info()
    print(f"\nModel Architecture:")
    print(f"  EEG output dim:    {model_info['eeg_output_dim']}")
    print(f"  Spec output dim:   {model_info['spec_output_dim']}")
    print(f"  Fusion output dim: {model_info['fusion_output_dim']}")
    print(f"  Num classes:       {model_info['num_classes']}")
    print(f"  Total parameters:  {model_info['total_params']:,}")
    print(f"  Trainable params:  {model_info['trainable_params']:,}\n")
    
    # WandB Logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        save_dir="logs",
        log_model=True,  # Log model checkpoints to WandB
    )
    
    # Log configuration to WandB
    wandb_logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))
    
    # Callbacks
    callbacks = []
    
    # Model Checkpoint - save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="hms-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early Stopping
    if config.training.early_stopping.get('patience'):
        early_stop_callback = EarlyStopping(
            monitor=config.training.early_stopping.monitor,
            patience=config.training.early_stopping.patience,
            mode=config.training.early_stopping.mode,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Trainer
    trainer = Trainer(
        max_epochs=config.training.num_epochs,
        accelerator="auto",  # Auto-detect GPU/CPU/MPS
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=16 if config.hardware.mixed_precision else 32,
        gradient_clip_val=1.0,  # Clip gradients to prevent exploding gradients
        log_every_n_steps=10,
        deterministic=False,  # Set to True for reproducibility (slower)
    )
    
    # Train
    print("Starting training...\n")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=resume_from_checkpoint,
    )
    
    # Test on best model
    print("\nTesting best model...\n")
    trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path="best",
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"WandB run: {wandb_logger.experiment.url}")
    print("="*60 + "\n")
    
    return trainer, model, datamodule


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train HMS Multi-Modal GNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="hms-brain-activity",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="WandB run name",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        resume_from_checkpoint=args.resume,
    )
