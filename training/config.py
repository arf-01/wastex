"""
Training configuration and paths.

All tuneable settings live here so they are easy to find, review,
and override without touching training logic.

Directory conventions
---------------------
::

    wastex/
    ├── models/
    │   ├── logits_mdl.keras          ← Original shipped model
    │   ├── classes.txt               ← Current class list
    │   └── versions/                 ← Versioned model artefacts
    │       └── model_v1_20260224…/
    │           ├── model.keras       ← Final saved model
    │           ├── best_model.keras  ← Best checkpoint (val_loss)
    │           ├── classes.txt       ← Class list for this model
    │           ├── metrics.json      ← Evaluation results
    │           ├── comparison.json   ← Delta vs previous model
    │           ├── config.json       ← Training config snapshot
    │           ├── training_log.json ← Loss / accuracy per epoch
    │           ├── training_log.csv  ← Same, CSV for tooling
    │           └── model_summary.txt ← Architecture summary
    │
    ├── datasets/                     ← Dataset versions (on disk)
    │   └── v1/
    │       ├── dataset_train/
    │       ├── dataset_test/
    │       └── dataset_val/
    │
    └── training/                     ← This package
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from django.conf import settings

# ── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR: Path = Path(settings.BASE_DIR)
DATASETS_ROOT: Path = BASE_DIR / "datasets"
MODELS_ROOT: Path = BASE_DIR / "models"
MODEL_VERSIONS_DIR: Path = MODELS_ROOT / "versions"

# The original shipped model (used as fallback for fine-tuning)
ORIGINAL_MODEL_PATH: Path = MODELS_ROOT / "logits_mdl.keras"
ORIGINAL_CLASSES_PATH: Path = MODELS_ROOT / "classes.txt"

# InceptionV3 ImageNet weights (notop) — used when training from scratch
INCEPTION_WEIGHTS_PATH: Path = MODELS_ROOT / "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"


@dataclass
class TrainingConfig:
    """All hyperparameters and settings for a single training run.

    Two-stage training (matching the original Kaggle pipeline):

    Stage 1 — FC head training:
        Freeze entire InceptionV3 base. Train only the classification
        head (GlobalAveragePooling2D → Dense 1024 → Dropout → Dense N).
        Uses ``learning_rate_fc`` (default 1e-4).

    Stage 2 — Late-layer fine-tuning:
        Unfreeze the last ``unfreeze_layers`` layers of the base
        (skipping BatchNorm if ``skip_bn_unfreeze`` is True).
        Uses ``learning_rate_ft`` (default 1e-5).

    Both stages use early stopping on ``val_loss`` and checkpoint
    the best model on ``val_accuracy``.

    Attributes
    ----------
    dataset_version : str
        Name of the ``DatasetVersion`` to train on (e.g. ``"v2"``).
    use_active : bool
        If True and ``dataset_version`` is empty, use the active version.
    base_model : str | None
        * ``None``      → fine-tune from the current shipped model.
        * ``"scratch"``  → fresh InceptionV3 from ImageNet weights.
        * ``"/path/…"`` → fine-tune from a specific checkpoint.
    epochs : int
        Max epochs per stage (default 40).
    batch_size : int
        Mini-batch size (default 8).
    learning_rate_fc : float
        LR for stage 1 — FC head (default 1e-4).
    learning_rate_ft : float
        LR for stage 2 — fine-tuning (default 1e-5).
    unfreeze_layers : int
        Number of base layers to unfreeze in stage 2 (default 60).
    skip_bn_unfreeze : bool
        Keep BatchNorm layers frozen during fine-tuning (default True).
    early_stopping_patience : int
        Stop if val_loss doesn't improve for this many epochs (default 8).
    """

    # ── Dataset ─────────────────────────────────────────────────────────
    dataset_version: str = ""
    use_active: bool = True

    # ── Base model ──────────────────────────────────────────────────────
    base_model: Optional[str] = None  # None → active model, "scratch" → ImageNet

    # ── Hyperparameters (matching original Kaggle pipeline) ─────────────
    epochs: int = 40
    batch_size: int = 8
    learning_rate_fc: float = 1e-4     # Stage 1: FC head only
    learning_rate_ft: float = 1e-5     # Stage 2: fine-tune last layers
    image_size: tuple = (299, 299)     # InceptionV3 default

    # ── Split names (must match folder names in dataset) ────────────────
    train_split_name: str = "dataset_train"
    validation_split_name: str = "dataset_val"
    test_split_name: str = "dataset_test"

    # ── Fine-tuning strategy ────────────────────────────────────────────
    unfreeze_layers: int = 60          # Unfreeze last N layers in stage 2
    skip_bn_unfreeze: bool = True      # Keep BatchNorm frozen during fine-tune

    # ── Early stopping ──────────────────────────────────────────────────
    early_stopping_patience: int = 8

    # ── Auto-promotion ──────────────────────────────────────────────────
    auto_promote: bool = False
    promotion_metric: str = "val_accuracy"

    # ── Metadata ────────────────────────────────────────────────────────
    notes: str = ""

    # ── Helpers ──────────────────────────────────────────────────────────

    def resolve_dataset_version(self) -> str:
        """Return the dataset version name to use for this run.

        Falls back to the active version if ``dataset_version`` is empty
        and ``use_active`` is True.

        Raises
        ------
        ValueError
            If no version can be resolved.
        """
        if self.dataset_version:
            return self.dataset_version

        if self.use_active:
            from classifier.models import DatasetVersion
            active = DatasetVersion.get_active()
            if active:
                return active.name

        raise ValueError(
            "No dataset version specified and no active version found. "
            "Pass --dataset <name> or activate a version first."
        )

    def model_output_dir(self, run_name: str) -> Path:
        """Return (and create) the output directory for a named run."""
        out = MODEL_VERSIONS_DIR / run_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (for saving alongside artefacts)."""
        return {
            "dataset_version": self.dataset_version,
            "use_active": self.use_active,
            "base_model": self.base_model,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate_fc": self.learning_rate_fc,
            "learning_rate_ft": self.learning_rate_ft,
            "image_size": list(self.image_size),
            "train_split_name": self.train_split_name,
            "validation_split_name": self.validation_split_name,
            "test_split_name": self.test_split_name,
            "unfreeze_layers": self.unfreeze_layers,
            "skip_bn_unfreeze": self.skip_bn_unfreeze,
            "early_stopping_patience": self.early_stopping_patience,
            "auto_promote": self.auto_promote,
            "promotion_metric": self.promotion_metric,
            "notes": self.notes,
        }
