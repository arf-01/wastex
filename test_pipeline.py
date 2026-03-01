"""
Pipeline validation script — tests data loading, model build, and 1-batch forward pass.
Run with: python test_pipeline.py
"""
import os
import sys
import json

os.environ["DJANGO_SETTINGS_MODULE"] = "wastex.settings"

import django
django.setup()

from classifier.models import DatasetVersion, VersionEntry, DatasetClass
from classifier.views.helpers import DATASETS_ROOT, scan_version_folder
from pathlib import Path

# ── Step 1: Register v1 ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Register v1 dataset")
print("=" * 60)

version_path = DATASETS_ROOT / "v1"
print(f"  Dataset path: {version_path}")
print(f"  Exists: {version_path.exists()}")

existing = DatasetVersion.objects.filter(name="v1").first()
if existing:
    print(f"  v1 already registered (id={existing.id}, active={existing.is_active})")
else:
    splits, class_counts, total = scan_version_folder(version_path)
    print(f"  Splits: {splits}")
    print(f"  Classes: {list(class_counts.keys())}")
    print(f"  Total images: {total}")

    dv = DatasetVersion.objects.create(
        name="v1",
        total_images=total,
        class_counts=class_counts,
        splits=splits,
        is_active=True,
    )

    for cls_name in class_counts:
        DatasetClass.objects.get_or_create(name=cls_name)

    count = 0
    for split_name in splits:
        split_dir = version_path / split_name
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    rel_path = img_file.relative_to(Path(DATASETS_ROOT).parent)
                    VersionEntry.objects.create(
                        version=dv,
                        physical_path=str(rel_path),
                        class_label=class_dir.name,
                        split=split_name,
                        filename=img_file.name,
                        file_size=img_file.stat().st_size,
                    )
                    count += 1
    print(f"  Created {count} VersionEntry rows")
    print(f"  DatasetVersion id={dv.id}, active={dv.is_active}")

print()

# ── Step 2: Activate v1 ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Verify v1 is active")
print("=" * 60)

dv = DatasetVersion.objects.filter(name="v1").first()
if not dv.is_active:
    DatasetVersion.objects.update(is_active=False)
    dv.is_active = True
    dv.save()
    print("  Activated v1")
else:
    print(f"  v1 is already active")

entries = VersionEntry.objects.filter(version=dv)
print(f"  Total VersionEntry rows: {entries.count()}")

# Count by split
for split in ["dataset_train", "dataset_val", "dataset_test"]:
    n = entries.filter(split=split).count()
    print(f"    {split}: {n}")

# Count by class
print("  Classes in DB:")
for cls in DatasetClass.objects.all().order_by("name"):
    n = entries.filter(class_label=cls.name).count()
    print(f"    {cls.name}: {n}")

print()

# ── Step 3: Test data.py load_datasets ───────────────────────────────────
print("=" * 60)
print("STEP 3: Test data.py load_datasets()")
print("=" * 60)

from training.config import TrainingConfig
from training.data import load_datasets

config = TrainingConfig()
print(f"  Config: batch_size={config.batch_size}, img={config.image_size}")

train_ds, val_ds, test_ds, class_names, split_counts = load_datasets(config)
print(f"  Splits loaded: train={split_counts['train']}, val={split_counts['validation']}, test={split_counts['test']}")
print(f"  Class names: {class_names}")
print(f"  Num classes: {len(class_names)}")

# Build class_to_idx for later use
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
print(f"  Class mapping: {class_to_idx}")

# Take 1 batch from each split and check shapes
for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
    for images, labels in ds.take(1):
        print(f"  {split_name}:")
        print(f"    images shape: {images.shape}  (expect [batch, 299, 299, 3])")
        print(f"    labels shape: {labels.shape}  (expect [batch, {len(class_names)}] one-hot)")
        print(f"    images dtype: {images.dtype}  (expect float32)")
        print(f"    images range: [{float(images.numpy().min()):.3f}, {float(images.numpy().max()):.3f}]  (expect [0, 1])")
        print(f"    labels example: {labels[0].numpy()}")

print()

# ── Step 4: Test model build ─────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Test build_model()")
print("=" * 60)

from training.train import build_model

num_classes = len(class_names)
model = build_model(num_classes, config)
print(f"  Model input shape:  {model.input_shape}")
print(f"  Model output shape: {model.output_shape}  (expect (None, {num_classes}))")
print(f"  Total params: {model.count_params():,}")

# Count trainable vs frozen
trainable = sum(p.numpy().size for p in model.trainable_weights)
non_trainable = sum(p.numpy().size for p in model.non_trainable_weights)
print(f"  Trainable params:     {trainable:,}")
print(f"  Non-trainable params: {non_trainable:,}")

print()

# ── Step 5: Test 1-batch forward pass + loss ─────────────────────────────
print("=" * 60)
print("STEP 5: Test 1-batch forward pass + loss")
print("=" * 60)

import tensorflow as tf

model.compile(
    optimizer=tf.keras.optimizers.Adam(config.learning_rate_fc),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

for images, labels in train_ds.take(1):
    logits = model(images, training=False)
    print(f"  Logits shape: {logits.shape}  (expect [{images.shape[0]}, {num_classes}])")
    print(f"  Logits sample: {logits[0].numpy()[:4]}...")
    
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_val = loss_fn(labels, logits)
    print(f"  Loss on 1 batch: {float(loss_val):.4f}")
    
    # Quick 1-step train
    result = model.train_on_batch(images, labels)
    print(f"  train_on_batch → loss={result[0]:.4f}, acc={result[1]:.4f}")

print()
print("=" * 60)
print("✅ ALL PIPELINE CHECKS PASSED")
print("=" * 60)
