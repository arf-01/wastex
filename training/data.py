"""
Dataset loading from the delta-based VersionEntry table.

Instead of reading from a physical folder tree, this module queries the
database for all images belonging to a given dataset version and split,
then builds ``tf.data.Dataset`` pipelines from the resolved physical
paths.  This means training always operates on exactly the images that
are registered in the version — no stale files, no missing entries.

Public API
----------
get_class_names  – Sorted, deduplicated class labels for a version.
load_split       – One split (train / val / test) → batched ``tf.data.Dataset``.
load_datasets    – All three splits in one call (convenience wrapper).

Usage::

    from training.config import TrainingConfig
    from training.data   import load_datasets

    config = TrainingConfig(dataset_version="v2")   # or leave blank → active
    train_ds, val_ds, test_ds, class_names, counts = load_datasets(config)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from django.conf import settings

from classifier.models import DatasetVersion, VersionEntry
from .config import TrainingConfig

logger = logging.getLogger(__name__)

BASE_DIR = Path(settings.BASE_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_class_names(version_name: str) -> list[str]:
    """Return sorted class labels present in a dataset version.

    Parameters
    ----------
    version_name : str
        The dataset version name, e.g. ``"v2"``.

    Returns
    -------
    list[str]
        Sorted, deduplicated class labels.

    Raises
    ------
    ValueError
        If the version has no entries at all.
    """
    labels = list(
        VersionEntry.objects
        .filter(version__name=version_name)
        .values_list("class_label", flat=True)
        .distinct()
        .order_by("class_label")
    )
    if not labels:
        raise ValueError(
            f"No class labels found for dataset version '{version_name}'. "
            f"Is it registered and populated?"
        )
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# Single-split loader
# ═══════════════════════════════════════════════════════════════════════════

def load_split(
    version_name: str,
    split_name: str,
    config: TrainingConfig,
    class_to_idx: dict[str, int],
    *,
    augment: bool = False,
) -> tuple[tf.data.Dataset, int]:
    """Load one split as a batched ``tf.data.Dataset``.

    The pipeline:
        1. Query ``VersionEntry`` for all paths + labels in this split.
        2. Filter out any entries whose physical file is missing.
        3. Build a ``tf.data`` pipeline with decode → resize → normalise.
        4. Optionally apply data augmentation (training only).
        5. Batch + prefetch.

    Parameters
    ----------
    version_name : str
        Dataset version name.
    split_name : str
        Split folder name (e.g. ``"dataset_train"``).
    config : TrainingConfig
        Training configuration (image size, batch size).
    class_to_idx : dict[str, int]
        Mapping from class label → integer index.
    augment : bool
        Whether to apply random augmentations (flip, brightness, etc.).

    Returns
    -------
    (tf.data.Dataset, int)
        The batched dataset and the total number of valid samples.

    Raises
    ------
    ValueError
        If the split contains zero usable images.
    """
    # ── 1. Query entries ────────────────────────────────────────────────
    entries = VersionEntry.objects.filter(
        version__name=version_name,
        split=split_name,
    ).values_list("physical_path", "class_label")

    paths: list[str] = []
    labels: list[int] = []
    skipped = 0

    for rel_path, class_label in entries:
        full_path = str(BASE_DIR / rel_path)
        if not Path(full_path).exists():
            skipped += 1
            continue
        paths.append(full_path)
        labels.append(class_to_idx[class_label])

    if skipped:
        logger.warning(
            "Split %s/%s: skipped %d entries (file not found on disk)",
            version_name, split_name, skipped,
        )

    total = len(paths)
    if total == 0:
        raise ValueError(
            f"No images found for {version_name}/{split_name}. "
            f"Is the dataset registered?"
        )

    logger.info(
        "Split %s/%s: %d images, %d classes",
        version_name, split_name, total, len(class_to_idx),
    )

    # ── 2. Build tf.data pipeline ───────────────────────────────────────
    h, w = config.image_size
    num_classes = len(class_to_idx)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if augment:
        ds = ds.shuffle(buffer_size=min(total, 10_000), seed=42)

    def _parse(file_path: tf.Tensor, label: tf.Tensor):
        """Read, decode, resize, normalise — return one-hot label."""
        raw = tf.io.read_file(file_path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [h, w])
        img = tf.cast(img, tf.float32) / 255.0
        label_onehot = tf.one_hot(label, depth=num_classes)
        return img, label_onehot

    def _augment(img: tf.Tensor, label: tf.Tensor):
        """Random augmentations for the training split."""
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, total


# ═══════════════════════════════════════════════════════════════════════════
# All-splits loader (convenience)
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(
    config: TrainingConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str], dict]:
    """Load train, validation, and test splits for a training run.

    If ``config.dataset_version`` is empty the **active** version is
    used automatically (``config.use_active`` must be True).

    Parameters
    ----------
    config : TrainingConfig
        Full training configuration.

    Returns
    -------
    (train_ds, val_ds, test_ds, class_names, split_counts)
        * Three batched ``tf.data.Dataset`` objects.
        * Sorted list of class names.
        * ``{"train": N, "validation": N, "test": N}`` sample counts.

    Raises
    ------
    ValueError
        If the version doesn't exist or has no entries.
    """
    # ── Resolve version ─────────────────────────────────────────────────
    version_name = config.resolve_dataset_version()

    if not DatasetVersion.objects.filter(name=version_name).exists():
        raise ValueError(f"Dataset version '{version_name}' not found in DB.")

    # ── Class mapping ───────────────────────────────────────────────────
    class_names = get_class_names(version_name)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    logger.info(
        "Dataset '%s': %d classes — %s",
        version_name, len(class_names), class_names,
    )

    # ── Load each split ─────────────────────────────────────────────────
    train_ds, n_train = load_split(
        version_name, config.train_split_name, config, class_to_idx,
        augment=True,
    )
    val_ds, n_val = load_split(
        version_name, config.validation_split_name, config, class_to_idx,
        augment=False,
    )
    test_ds, n_test = load_split(
        version_name, config.test_split_name, config, class_to_idx,
        augment=False,
    )

    split_counts = {
        "train": n_train,
        "validation": n_val,
        "test": n_test,
    }

    logger.info(
        "Splits loaded — train=%d, val=%d, test=%d  (total %d)",
        n_train, n_val, n_test, sum(split_counts.values()),
    )

    return train_ds, val_ds, test_ds, class_names, split_counts
