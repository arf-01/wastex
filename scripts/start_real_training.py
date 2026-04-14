"""
Launch a REAL training run — 5 epochs, both stages, full dataset.
Old model (models/logits_mdl.keras) is NOT deleted.
New model saved under models/versions/<run_name>/model.keras

Run with: python start_real_training.py
"""
import os, sys, warnings
os.environ["DJANGO_SETTINGS_MODULE"] = "wastex.settings"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import django; django.setup()

from training.config import TrainingConfig
from training.runner import run_training
from pathlib import Path

# Verify old model is safe
old_model = Path("models/logits_mdl.keras")
print(f"Old model exists: {old_model.exists()} — will NOT be touched")
print()

config = TrainingConfig(
    dataset_version="v1",
    epochs=5,
    batch_size=8,
    early_stopping_patience=5,   # won't early-stop with only 5 epochs
    unfreeze_layers=60,
    skip_bn_unfreeze=True,
    auto_promote=False,          # don't auto-promote, let user decide
)

print("=" * 60)
print("STARTING FULL TRAINING RUN")
print("=" * 60)
print(f"  Dataset version : v1")
print(f"  Epochs          : {config.epochs}")
print(f"  Batch size      : {config.batch_size}")
print(f"  Unfreeze layers : {config.unfreeze_layers}")
print(f"  Auto promote    : {config.auto_promote}")
print("=" * 60)
print()

run = run_training(config)

print()
print("=" * 60)
print(f"RUN COMPLETE: {run.run_name}")
print(f"  Status     : {run.status}")
print(f"  Accuracy   : {run.test_accuracy}")
print(f"  F1 Score   : {run.test_f1}")
print(f"  Model saved: {run.model_path}")
print(f"  Old model  : {old_model} (untouched, exists={old_model.exists()})")
print("=" * 60)
