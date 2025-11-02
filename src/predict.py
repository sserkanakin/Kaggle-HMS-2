"""Inference script to run predictions with a trained checkpoint.

Loads model config, constructs the LightningModule, loads checkpoint weights,
builds a test split from processed data using HMSDataModule, and writes a CSV
with predictions per sample (patient_id, label_id, probabilities).
"""

from __future__ import annotations

import sys
from pathlib import Path
import csv
import time
import torch
from torch.nn.functional import softmax
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import HMSDataModule
from src.lightning_trainer import HMSLightningModule


@torch.no_grad()
def predict(
	config_path: str = "configs/model.yaml",
	checkpoint_path: str = "checkpoints/last.ckpt",
	output_csv: str = "predictions.csv",
	batch_size_override: int | None = None,
	num_workers_override: int | None = None,
	max_batches: int | None = None,
):
	"""Run inference over the test split and save predictions.

	Note: For now, 'test' is the held-out split from processed training data.
	"""
	config = OmegaConf.load(config_path)

	# Build DataModule (use test split)
	dm = HMSDataModule(
		data_dir=config.data.get('data_dir', 'data/processed'),
		train_csv=config.data.get('train_csv', 'data/raw/train_unique.csv'),
		batch_size=(batch_size_override if batch_size_override is not None else config.training.batch_size),
		n_folds=config.data.get('n_folds', 5),
		current_fold=config.data.get('current_fold', 0),
		stratify_by_evaluators=config.data.get('stratify_by_evaluators', True),
		evaluator_bins=list(config.data.get('evaluator_bins', [0, 5, 10, 15, 20, 999])),
		min_evaluators=config.data.get('min_evaluators', 0),
		num_workers=(num_workers_override if num_workers_override is not None else config.data.num_workers),
		pin_memory=config.data.pin_memory,
		shuffle_seed=config.data.shuffle_seed,
		compute_class_weights=False,
	)
	dm.setup(stage="test")

	# Build model and load checkpoint
	model = HMSLightningModule(
		model_config=config.model,
		num_classes=config.model.num_classes,
		learning_rate=config.training.learning_rate,
		weight_decay=config.training.weight_decay,
		class_weights=None,
		scheduler_config=config.training.scheduler,
	)
	# In torch>=2.6, torch.load defaults to weights_only=True which may fail for Lightning checkpoints
	ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	model.load_state_dict(ckpt["state_dict"], strict=False)
	model.eval()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	loader = dm.test_dataloader()

	# Logging: dataset/loader info
	try:
		total_batches = len(loader)
	except Exception:
		total_batches = None
	print("\n" + "="*60)
	print("Inference Start")
	print(f"  Device:        {device}")
	print(f"  Batch size:    {loader.batch_size}")
	print(f"  Num workers:   {loader.num_workers}")
	print(f"  Test samples:  {len(dm.test_dataset)}")
	if total_batches is not None:
		print(f"  Batches:       {total_batches}")
	if max_batches is not None:
		print(f"  Max batches:   {max_batches}")
	print("="*60 + "\n")

	# Write predictions
	out_path = Path(output_csv)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", newline="") as f:
		writer = csv.writer(f)
		header = ["patient_id", "label_id"] + [f"p_{i}" for i in range(dm.get_num_classes())]
		writer.writerow(header)

		batch_times: list[float] = []
		for bidx, batch in enumerate(loader, 1):
			start = time.time()
			eeg_graphs = batch['eeg_graphs']  # List[Batch]
			spec_graphs = batch['spec_graphs']  # List[Batch]
			patient_ids = batch['patient_ids']
			label_ids = batch['label_ids']

			# Move PyG Batches to device
			eeg_graphs = [g.to(device) for g in eeg_graphs]
			spec_graphs = [g.to(device) for g in spec_graphs]

			logits = model(eeg_graphs, spec_graphs)
			probs = softmax(logits, dim=-1).cpu().tolist()

			for pid, lid, pr in zip(patient_ids, label_ids, probs):
				writer.writerow([int(pid), int(lid)] + [float(x) for x in pr])

			# Batch timing/logging
			elapsed = time.time() - start
			batch_times.append(elapsed)
			if (bidx % 5 == 1) or (max_batches is not None) or (total_batches is not None and bidx in {1, total_batches}):
				avg = sum(batch_times) / len(batch_times)
				print(f"[Infer] Batch {bidx}{'/' + str(total_batches) if total_batches else ''} | {elapsed:.3f}s (avg {avg:.3f}s)")
			if max_batches is not None and bidx >= max_batches:
				print(f"[Infer] Reached max_batches={max_batches}, stopping early for testing.")
				break

	print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run inference with trained checkpoint")
	parser.add_argument("--config", type=str, default="configs/model.yaml", help="Model config path")
	parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path (.ckpt)")
	parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path")
	parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
	parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers")
	parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches for quick tests")
	args = parser.parse_args()

	predict(
		config_path=args.config,
		checkpoint_path=args.checkpoint,
		output_csv=args.output,
		batch_size_override=args.batch_size,
		num_workers_override=args.num_workers,
		max_batches=args.max_batches,
	)

