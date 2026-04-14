"""
Micro training run — validates the FULL pipeline end-to-end.

Uses only a few batches of data, 2 epochs, both stages.
Should finish in ~2-3 minutes on CPU.

Run with: python test_micro_train.py
"""
import os, sys, time, warnings
os.environ["DJANGO_SETTINGS_MODULE"] = "wastex.settings"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import django; django.setup()

import numpy as np
import tensorflow as tf
from pathlib import Path

from classifier.models import DatasetVersion, VersionEntry, TrainingRun
from training.config import TrainingConfig
from training.data import load_datasets
from training.train import build_model, train_stage1, train_stage2
from training.evaluate import evaluate_model, compare_with_previous

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MICRO TRAINING PIPELINE TEST")
print("2 epochs, both stages, tiny subset")
print("=" * 60)
start_time = time.time()

# ── 1. Verify v1 is active ──────────────────────────────────────────
print("\n--- Step 1: Dataset version ---")
dv = DatasetVersion.objects.filter(name="v1", is_active=True).first()
check("v1 active in DB", dv is not None)

# ── 2. Config with minimal epochs ───────────────────────────────────
print("\n--- Step 2: Config ---")
config = TrainingConfig(
    dataset_version="v1",
    epochs=2,                    # just 2 epochs per stage
    batch_size=8,
    early_stopping_patience=2,
    unfreeze_layers=10,          # fewer layers = faster
)
check("Config created", True, f"epochs=2, batch=8, unfreeze=10")

# ── 3. Load datasets ────────────────────────────────────────────────
print("\n--- Step 3: Load datasets ---")
train_ds, val_ds, test_ds, class_names, counts = load_datasets(config)

check("train_ds loaded", counts["train"] > 0, f"{counts['train']} images")
check("val_ds loaded", counts["validation"] > 0, f"{counts['validation']} images")
check("test_ds loaded", counts["test"] > 0, f"{counts['test']} images")
check("8 classes", len(class_names) == 8, str(class_names))

# Take only 3 batches from train/val to speed things up
train_ds_mini = train_ds.take(3)   # 3 batches * 8 = 24 images
val_ds_mini = val_ds.take(2)       # 2 batches * 8 = 16 images
test_ds_mini = test_ds.take(2)     # 2 batches for eval

# Verify shapes
for imgs, lbls in train_ds_mini.take(1):
    check("Image shape", imgs.shape[1:] == (299, 299, 3), str(imgs.shape))
    check("Label shape (one-hot)", lbls.shape[1] == 8, str(lbls.shape))
    check("Image range [0,1]", float(imgs.numpy().min()) >= 0 and float(imgs.numpy().max()) <= 1.0)
    check("Image dtype float32", imgs.dtype == tf.float32)

# ── 4. Build model ──────────────────────────────────────────────────
print("\n--- Step 4: Build model ---")
model = build_model(num_classes=len(class_names), config=config)
check("Model built", model is not None)
check("Input shape", model.input_shape == (None, 299, 299, 3), str(model.input_shape))
check("Output shape", model.output_shape == (None, 8), str(model.output_shape))

# Count trainable params (should be head only at this point)
trainable_count = sum(w.numpy().size for w in model.trainable_weights)
total_count = model.count_params()
check("Base frozen", trainable_count < total_count * 0.2,
      f"trainable={trainable_count:,} / total={total_count:,}")

# ── 5. Stage 1: FC head (2 epochs on mini data) ────────────────────
print("\n--- Step 5: Stage 1 - FC head training (2 epochs) ---")
output_dir = Path("models/versions/_test_micro_run")
output_dir.mkdir(parents=True, exist_ok=True)

t1 = time.time()
model = train_stage1(model, train_ds_mini, val_ds_mini, config, output_dir)
t1_elapsed = time.time() - t1

check("Stage 1 completed", model is not None, f"{t1_elapsed:.1f}s")
check("FC checkpoint saved", (output_dir / "fc_best.keras").exists())

# Verify model still outputs correct shape
for imgs, _ in train_ds_mini.take(1):
    logits = model(imgs, training=False)
    check("Post-stage1 logits shape", logits.shape == (imgs.shape[0], 8), str(logits.shape))

# ── 6. Stage 2: Fine-tune (2 epochs on mini data) ──────────────────
print("\n--- Step 6: Stage 2 - Fine-tune (2 epochs) ---")
t2 = time.time()
model = train_stage2(model, train_ds_mini, val_ds_mini, config, output_dir)
t2_elapsed = time.time() - t2

check("Stage 2 completed", model is not None, f"{t2_elapsed:.1f}s")
check("FT checkpoint saved", (output_dir / "ft_best.keras").exists())

# ── 7. Save final model ────────────────────────────────────────────
print("\n--- Step 7: Save model ---")
final_path = output_dir / "model.keras"
model.save(str(final_path))
check("Final model saved", final_path.exists(), f"{final_path.stat().st_size / 1024 / 1024:.1f} MB")

# ── 8. Evaluate on test (mini) ──────────────────────────────────────
print("\n--- Step 8: Evaluate ---")
metrics = evaluate_model(model, test_ds_mini, class_names, output_dir)

check("Accuracy computed", "accuracy" in metrics, f"acc={metrics.get('accuracy')}")
check("Precision computed", "precision" in metrics, f"prec={metrics.get('precision')}")
check("Recall computed", "recall" in metrics, f"rec={metrics.get('recall')}")
check("Macro F1 computed", "macro_f1" in metrics, f"f1={metrics.get('macro_f1')}")
check("Loss computed", "loss" in metrics, f"loss={metrics.get('loss')}")
check("Confusion matrix saved", (output_dir / "confusion_matrix.png").exists())
check("Classification report saved", (output_dir / "classification_report.txt").exists())
check("metrics.json saved", (output_dir / "metrics.json").exists())

# ── 9. Comparison (no previous) ────────────────────────────────────
print("\n--- Step 9: Comparison ---")
comparison = compare_with_previous(metrics, None, output_dir)
check("Comparison generated", "recommendation" in comparison, comparison.get("recommendation"))
check("comparison.json saved", (output_dir / "comparison.json").exists())

# ── 10. Cleanup test artefacts ──────────────────────────────────────
print("\n--- Step 10: Cleanup ---")
import shutil
shutil.rmtree(str(output_dir), ignore_errors=True)
check("Test artefacts cleaned", not output_dir.exists())

# ══════════════════════════════════════════════════════════════════════
elapsed = time.time() - start_time
print("\n" + "=" * 60)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
print(f"RESULTS: {passed} passed, {failed} failed  ({elapsed:.0f}s total)")

if failed:
    print("\nFAILED CHECKS:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  X {name}  {detail}")
    print("=" * 60)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — pipeline is ready for full training")
    print("=" * 60)
    sys.exit(0)
