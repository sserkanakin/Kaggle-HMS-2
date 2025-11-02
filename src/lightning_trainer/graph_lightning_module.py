"""PyTorch Lightning module for HMS Multi-Modal GNN."""

from __future__ import annotations

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from src.models import HMSMultiModalGNN


class HMSLightningModule(LightningModule):
    """Lightning wrapper for HMS Multi-Modal GNN.
    
    Handles:
    - Training, validation, and test steps
    - Loss computation with optional class weights
    - Metrics: accuracy, F1, precision, recall (macro & per-class)
    - WandB logging
    - Learning rate scheduling
    - Optimizer configuration
    
    Parameters
    ----------
    model_config : dict
        Model configuration (eeg_encoder, spec_encoder, fusion, classifier)
    num_classes : int
        Number of output classes (default: 6)
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        Weight decay for optimizer
    class_weights : torch.Tensor, optional
        Class weights for loss balancing
    scheduler_config : dict, optional
        Learning rate scheduler configuration
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        loss: str | None = None,
    ) -> None:
        super().__init__()
        
        # Save hyperparameters (will be logged to WandB)
        self.save_hyperparameters(ignore=['class_weights'])
        
        # Initialize model
        self.model = HMSMultiModalGNN(
            eeg_config=model_config.get('eeg_encoder'),
            spec_config=model_config.get('spec_encoder'),
            fusion_config=model_config.get('fusion'),
            classifier_config=model_config.get('classifier'),
            num_classes=num_classes,
        )
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        self.loss_type = (loss or 'kl').lower()  # 'kl' or 'cross_entropy'
        
        # Loss function
        if self.loss_type == 'cross_entropy':
            if class_weights is not None:
                self.register_buffer('class_weights', class_weights)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.class_weights = None
                self.criterion = nn.CrossEntropyLoss()
        else:
            # KL divergence between target probability distribution and predicted probabilities
            # We'll compute via F.kl_div(log_softmax(logits), target_probs, reduction='batchmean')
            self.class_weights = None
            self.criterion = None  # handled inline
        
        # Metrics for each stage
        metrics = MetricCollection({
            'acc': Accuracy(task='multiclass', num_classes=num_classes, average='micro'),
            'acc_macro': Accuracy(task='multiclass', num_classes=num_classes, average='macro'),
            'f1_macro': MulticlassF1Score(num_classes=num_classes, average='macro'),
            'precision_macro': MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall_macro': MulticlassRecall(num_classes=num_classes, average='macro'),
        })
        
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        
        # Per-class accuracy (for detailed analysis)
        self.train_acc_per_class = Accuracy(
            task='multiclass', num_classes=num_classes, average='none'
        )
        self.val_acc_per_class = Accuracy(
            task='multiclass', num_classes=num_classes, average='none'
        )
        self.test_acc_per_class = Accuracy(
            task='multiclass', num_classes=num_classes, average='none'
        )
    
    def forward(self, eeg_graphs, spec_graphs):
        """Forward pass through the model."""
        return self.model(eeg_graphs, spec_graphs)
    
    def _shared_step(self, batch: Dict, batch_idx: int, stage: str):
        """Shared step for train/val/test.
        
        Parameters
        ----------
        batch : dict
            Batch from dataloader
        batch_idx : int
            Batch index
        stage : str
            One of 'train', 'val', 'test'
        """
        eeg_graphs = batch['eeg_graphs']
        spec_graphs = batch['spec_graphs']
        targets = batch['targets']
        target_probs = batch.get('target_probs', None)
        
        # Forward pass
        logits = self(eeg_graphs, spec_graphs)
        
        # Compute loss
        if self.loss_type == 'kl':
            if target_probs is None:
                raise RuntimeError("Batch missing 'target_probs' required for KL loss")
            log_probs = F.log_softmax(logits, dim=1)
            # reduction='batchmean' matches Kaggle metric scaling
            loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
        else:
            loss = self.criterion(logits, targets)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        if stage == 'train':
            metrics = self.train_metrics(preds, targets)
            self.train_acc_per_class(preds, targets)
        elif stage == 'val':
            metrics = self.val_metrics(preds, targets)
            self.val_acc_per_class(preds, targets)
        else:  # test
            metrics = self.test_metrics(preds, targets)
            self.test_acc_per_class(preds, targets)
        
        # Log loss (provide batch_size to avoid ambiguous batch size warning)
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        
        # Log metrics with explicit batch_size to avoid ambiguous batch size warning
        for k, v in metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        
        return loss
    
    def training_step(self, batch: Dict, batch_idx: int):
        """Training step."""
        return self._shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step."""
        return self._shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch: Dict, batch_idx: int):
        """Test step."""
        return self._shared_step(batch, batch_idx, 'test')

    def transfer_batch_to_device(self, batch: Dict[str, Any], device: torch.device, dataloader_idx: int) -> Dict[str, Any]:
        """Move PyG Batch objects and tensors to the target device.
        Lightning doesn't automatically move custom objects like torch_geometric Batch.
        """
        batch = dict(batch)  # shallow copy
        if 'eeg_graphs' in batch:
            batch['eeg_graphs'] = [g.to(device, non_blocking=True) for g in batch['eeg_graphs']]
        if 'spec_graphs' in batch:
            batch['spec_graphs'] = [g.to(device, non_blocking=True) for g in batch['spec_graphs']]
        if 'targets' in batch and isinstance(batch['targets'], torch.Tensor):
            batch['targets'] = batch['targets'].to(device, non_blocking=True)
        if 'target_probs' in batch and isinstance(batch['target_probs'], torch.Tensor):
            batch['target_probs'] = batch['target_probs'].to(device, non_blocking=True)
        return batch
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Compute and log per-class accuracy
        per_class_acc = self.train_acc_per_class.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f'train/acc_class_{i}', acc, prog_bar=False)
        self.train_acc_per_class.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Compute and log per-class accuracy
        per_class_acc = self.val_acc_per_class.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f'val/acc_class_{i}', acc, prog_bar=False)
        self.val_acc_per_class.reset()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Compute and log per-class accuracy
        per_class_acc = self.test_acc_per_class.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f'test/acc_class_{i}', acc, prog_bar=False)
        self.test_acc_per_class.reset()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Return optimizer only if no scheduler
        if not self.scheduler_config:
            return optimizer
        
        # Configure scheduler
        scheduler_type = self.scheduler_config.get('type', 'ReduceLROnPlateau')
        
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_config.get('mode', 'min'),
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 5),
                min_lr=self.scheduler_config.get('min_lr', 1e-6),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.scheduler_config.get('monitor', 'val/loss'),
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        elif scheduler_type == 'CosineAnnealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 50),
                eta_min=self.scheduler_config.get('min_lr', 1e-6),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        elif scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get('step_size', 10),
                gamma=self.scheduler_config.get('gamma', 0.1),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        else:
            # Unknown scheduler, return optimizer only
            print(f"Warning: Unknown scheduler type '{scheduler_type}', using no scheduler")
            return optimizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return self.model.get_model_info()


__all__ = ["HMSLightningModule"]
