"""
Training run orchestrator — ties data → train → evaluate → save.

This is the main entry point for a complete retraining cycle:

1. Resolve the dataset version and load train / val / test splits.
2. Build the InceptionV3 model.
3. Stage 1: train FC head (base frozen).
4. Stage 2: fine-tune last 60 layers (skip BN).
5. Evaluate on test split → metrics, confusion matrix, classification report.
6. Compare against previous best model.
7. Save all artefacts under ``models/versions/<run_name>/``.
8. Optionally promote the new model to active.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from django.utils import timezone

from classifier.models import TrainingRun
from .config import TrainingConfig, MODEL_VERSIONS_DIR
from .data import load_datasets
from .evaluate import compare_with_previous, evaluate_model
from .train import build_model, train_stage1, train_stage2

logger = logging.getLogger(__name__)


def _find_previous_metrics() -> Path | None:
    """Locate the most recent completed run's ``metrics.json``."""
    last_run = (
        TrainingRun.objects
        .filter(status="completed", model_path__gt="")
        .order_by("-finished_at")
        .first()
    )
    if last_run and last_run.model_path:
        metrics_path = Path(last_run.model_path).parent / "metrics.json"
        if metrics_path.exists():
            return metrics_path
    return None


def run_training(config: TrainingConfig) -> TrainingRun:
    """Execute a full two-stage training run end-to-end.

    Parameters
    ----------
    config : TrainingConfig
        All hyperparameters and settings.

    Returns
    -------
    TrainingRun
        The completed (or failed) database record.
    """
    version_name = config.resolve_dataset_version()

    # Generate unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"model_{version_name}_{timestamp}"
    output_dir = config.model_output_dir(run_name)

    # Create TrainingRun record
    run = TrainingRun.objects.create(
        run_name=run_name,
        dataset_version_name=version_name,
        status="pending",
        config=config.to_dict(),
    )

    try:
        # ── 1. Load data ────────────────────────────────────────────
        run.status = "running"
        run.save(update_fields=["status"])
        logger.info("Loading dataset version '%s'…", version_name)

        train_ds, val_ds, test_ds, class_names, split_counts = load_datasets(config)

        run.num_classes = len(class_names)
        run.num_train_samples = split_counts["train"]
        run.num_val_samples = split_counts["validation"]
        run.num_test_samples = split_counts["test"]
        run.save(update_fields=[
            "num_classes", "num_train_samples",
            "num_val_samples", "num_test_samples",
        ])

        # Save class list for this run
        (output_dir / "classes.txt").write_text(
            "\n".join(class_names) + "\n", encoding="utf-8",
        )

        # ── 2. Build model ──────────────────────────────────────────
        model = build_model(num_classes=len(class_names), config=config)

        # Save model summary
        summary_lines: list[str] = []
        model.summary(print_fn=lambda s: summary_lines.append(s))
        (output_dir / "model_summary.txt").write_text(
            "\n".join(summary_lines), encoding="utf-8",
        )

        # ── 3. Stage 1: FC head training ────────────────────────────
        run.status = "training"
        run.started_at = timezone.now()
        run.save(update_fields=["status", "started_at"])

        model = train_stage1(model, train_ds, val_ds, config, output_dir)

        # ── 4. Stage 2: Fine-tune last layers ──────────────────────
        model = train_stage2(model, train_ds, val_ds, config, output_dir)

        # Save final model
        final_path = output_dir / "model.keras"
        model.save(str(final_path))
        logger.info("Saved final model to %s", final_path)

        # ── 5. Evaluate on test split ───────────────────────────────
        run.status = "evaluating"
        run.save(update_fields=["status"])

        metrics = evaluate_model(model, test_ds, class_names, output_dir)

        run.test_accuracy = metrics["accuracy"]
        run.test_f1 = metrics["macro_f1"]
        run.metrics_summary = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "macro_f1": metrics["macro_f1"],
            "loss": metrics["loss"],
        }

        # ── 6. Compare with previous model ─────────────────────────
        prev_path = _find_previous_metrics()
        comparison = compare_with_previous(metrics, prev_path, output_dir)
        run.comparison = comparison

        # Save config snapshot
        (output_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2), encoding="utf-8",
        )

        # ── 7. Finalise run record ─────────────────────────────────
        run.model_path = str(final_path)
        run.status = "completed"
        run.finished_at = timezone.now()
        run.save(update_fields=[
            "test_accuracy", "test_f1", "metrics_summary",
            "comparison", "model_path", "status", "finished_at",
        ])

        # ── 8. Auto-promote if better ──────────────────────────────
        if config.auto_promote and comparison.get("recommendation") == "promote":
            run.promote()
            logger.info("Auto-promoted run '%s' to active model", run_name)

        logger.info(
            "═══ TRAINING COMPLETE: %s ═══\n"
            "  Accuracy : %.4f\n"
            "  Precision: %.4f\n"
            "  Recall   : %.4f\n"
            "  Macro F1 : %.4f\n"
            "  Artefacts: %s",
            run_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["macro_f1"],
            output_dir,
        )

    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error_message", "finished_at"])
        logger.exception("Training run '%s' failed", run_name)
        raise

    return run
