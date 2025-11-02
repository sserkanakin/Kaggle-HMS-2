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
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    resume_from_checkpoint: str | None = None,
    *,
    smoke: bool = False,
    offline: bool = False,
    limit_train_batches: int | float | None = None,
    limit_val_batches: int | float | None = None,
    max_epochs_override: int | None = None,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
    prefetch_factor_override: int | None = None,
    pin_memory_override: bool | None = None,
    mp_sharing_strategy: str = "auto",  # 'auto' | 'file_descriptor' | 'file_system'
    accumulate_grad_batches_override: int | None = None,
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
    
    # Environment & CUDA performance tuning
    # - Limit intra-op/OpenMP threads per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Enable fast math paths on NVIDIA Tensor Core GPUs
    try:
        torch.set_float32_matmul_precision('high')  # enables TF32 matmuls
        if torch.backends.cuda.is_built():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    except Exception as _e:
        print(f"[Warning] Could not enable TF32/cudnn benchmark: {_e}")

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

    # Choose multiprocessing sharing strategy to avoid /dev/shm mmap issues
    # 'file_descriptor' reduces reliance on shared filenames in /dev/shm
    try:
        effective_pin = pin_memory_override if pin_memory_override is not None else config.data.pin_memory
        if mp_sharing_strategy == 'file_descriptor' or (mp_sharing_strategy == 'auto' and effective_pin):
            _mp.set_sharing_strategy('file_descriptor')
            print('[Info] torch.multiprocessing sharing strategy set to file_descriptor')
        else:
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
        prefetch_factor=(prefetch_factor_override if prefetch_factor_override is not None else config.data.get('prefetch_factor', 4)),
        pin_memory=(pin_memory_override if pin_memory_override is not None else config.data.pin_memory),
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
        loss=str(config.training.get('loss', 'kl')),
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
    
    # WandB Logger with helpful defaults (name/group/tags)
    # Derive a descriptive default name if not provided
    try:
        current_fold = config.data.get('current_fold', 0)
        mixed_precision = (
            'bf16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
                '16' if (getattr(config, 'hardware', None) and getattr(config.hardware, 'mixed_precision', False)) else '32'
            )
        )
        default_name = f"hms-kl-{mixed_precision}-bs{dm_batch_size}-fold{current_fold}"
    except Exception:
        default_name = None
    run_name = wandb_name or default_name

    # Default tags if none provided
    default_tags = [
        f"fold{config.data.get('current_fold', 0)}",
        f"loss={str(config.training.get('loss','kl'))}",
        f"mp={'bf16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else ('16' if (getattr(config, 'hardware', None) and getattr(config.hardware, 'mixed_precision', False)) else '32')}",
        f"bs={dm_batch_size}",
        f"nw={dm_num_workers}",
        f"pf={prefetch_factor_override if prefetch_factor_override is not None else config.data.get('prefetch_factor', 4)}",
        f"pin={'on' if (pin_memory_override if pin_memory_override is not None else config.data.pin_memory) else 'off'}",
    ]
    tags = wandb_tags or default_tags

    wandb_logger = WandbLogger(
        project=wandb_project,
        name=run_name,
        save_dir="logs",
        log_model=True,
        group=wandb_group,
        tags=tags,
    )
    
    # Log configuration to WandB
    wandb_logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))
    # Also log runtime overrides for traceability
    wandb_logger.experiment.config.update({
        'runtime': {
            'batch_size_override': batch_size_override,
            'num_workers_override': num_workers_override,
            'prefetch_factor_override': prefetch_factor_override,
            'pin_memory_override': pin_memory_override,
            'max_epochs_override': max_epochs_override,
        }
    })
    
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
        precision=(
            ('bf16-mixed' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else '16-mixed')
            if getattr(config, 'hardware', None) and getattr(config.hardware, 'mixed_precision', False) else 32
        ),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=False,
        accumulate_grad_batches=(accumulate_grad_batches_override if accumulate_grad_batches_override is not None else getattr(config.training, 'accumulate_grad_batches', 1)),
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
        "--wandb-group",
        type=str,
        default=None,
        help="WandB run group (to cluster related runs)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated list of W&B tags (e.g. fold0,kl,bf16,bs16)",
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
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Override prefetch_factor for DataModule (only used when num_workers>0)",
    )
    parser.add_argument(
        "--disable-pin-memory",
        action="store_true",
        help="Disable pin_memory in DataLoaders to reduce host memory pressure",
    )
    parser.add_argument(
        "--mp-sharing-strategy",
        type=str,
        default="auto",
        choices=["auto", "file_descriptor", "file_system"],
        help="Multiprocessing sharing strategy to use for PyTorch (auto prefers file_descriptor when pin_memory is enabled)",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=None,
        help="Accumulate gradients over N batches to emulate larger global batch without increasing per-step memory",
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        wandb_project=args.wandb_project,
    wandb_name=args.wandb_name,
    wandb_group=args.wandb_group,
    wandb_tags=(args.wandb_tags.split(',') if args.wandb_tags else None),
        resume_from_checkpoint=args.resume,
        smoke=args.smoke,
        offline=args.offline,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        max_epochs_override=args.max_epochs,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        prefetch_factor_override=args.prefetch_factor,
        pin_memory_override=(False if args.disable_pin_memory else None),
        mp_sharing_strategy=args.mp_sharing_strategy,
        accumulate_grad_batches_override=args.accumulate_grad_batches,
    )
