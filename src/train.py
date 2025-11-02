"""Main training script for HMS Multi-Modal GNN."""

import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import os
import resource
import torch.multiprocessing as _mp

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
    *,
    smoke: bool = False,
    offline: bool = False,
    limit_train_batches: int | float | None = None,
    limit_val_batches: int | float | None = None,
    max_epochs_override: int | None = None,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
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

    # Smoke/offline adjustments
    if offline or smoke:
        import os as _os
        _os.environ.setdefault("WANDB_MODE", "offline")
        print("[Info] WANDB_MODE=offline (no internet required)")
    
    # Environment & multiprocessing tuning to avoid 'Too many open files' and shared-memory errors
    # - Limit intra-op/OpenMP threads per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Try to raise the soft RLIMIT_NOFILE if permitted (helps with many shared-memory fds)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 65536
        new_soft = min(target, hard) if hard != resource.RLIM_INFINITY else target
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            print(f"[Info] Increased RLIMIT_NOFILE: {soft} -> {new_soft} (hard={hard})")
        else:
            print(f"[Info] RLIMIT_NOFILE soft={soft}, hard={hard} (no change)")
    except Exception as e:
        print(f"[Warning] Could not increase RLIMIT_NOFILE: {e}")

    # Prefer file_system sharing strategy to reduce open fd pressure on some systems
    try:
        _mp.set_sharing_strategy('file_system')
        print('[Info] torch.multiprocessing sharing strategy set to file_system')
    except Exception as e:
        print(f"[Warning] Could not set torch.multiprocessing sharing strategy: {e}")
    # Initialize DataModule
    print("Initializing DataModule...")
    dm_batch_size = batch_size_override if batch_size_override is not None else config.training.batch_size
    dm_num_workers = num_workers_override if num_workers_override is not None else config.data.num_workers
    if smoke:
        dm_batch_size = min(2, dm_batch_size)
        dm_num_workers = 0

    datamodule = HMSDataModule(
        data_dir=config.data.get('data_dir', 'data/processed'),
        train_csv=config.data.get('train_csv', 'data/raw/train_unique.csv'),
        batch_size=dm_batch_size,
        n_folds=config.data.get('n_folds', 5),
        current_fold=config.data.get('current_fold', 0),
        stratify_by_evaluators=config.data.get('stratify_by_evaluators', True),
        evaluator_bins=list(config.data.get('evaluator_bins', [0, 5, 10, 15, 20, 999])),
        min_evaluators=config.data.get('min_evaluators', 0),
        num_workers=dm_num_workers,
        pin_memory=config.data.pin_memory,
        shuffle_seed=config.data.shuffle_seed,
        compute_class_weights=(not smoke),
    )
    
    # Setup datasets; optionally get class weights
    datamodule.setup(stage="fit")
    class_weights = datamodule.get_class_weights() if (not smoke and config.training.use_class_weights) else None
    
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
        log_model=True,
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
        max_epochs=(max_epochs_override if max_epochs_override is not None else (1 if smoke else config.training.num_epochs)),
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=16 if getattr(config, 'hardware', None) and getattr(config.hardware, 'mixed_precision', False) else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=False,
        limit_train_batches=(limit_train_batches if limit_train_batches is not None else (2 if smoke else 1.0)),
        limit_val_batches=(limit_val_batches if limit_val_batches is not None else (1 if smoke else 1.0)),
        num_sanity_val_steps=(0 if smoke else 2),
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
    # Ensure test split is prepared (uses validation fold by default)
    datamodule.setup(stage="test")
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
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast smoke test (1 epoch, 2 train batches, 1 val batch, WANDB offline)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force WANDB offline mode",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=None,
        help="Override limit_train_batches for Trainer (float fraction or int)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=None,
        help="Override limit_val_batches for Trainer (float fraction or int)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs for Trainer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for DataModule",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override num_workers for DataModule",
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        resume_from_checkpoint=args.resume,
        smoke=args.smoke,
        offline=args.offline,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        max_epochs_override=args.max_epochs,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
    )
