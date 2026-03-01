"""
Test evaluation — mirrors the Kaggle notebook's evaluation section.

Produces:
- Test accuracy, precision, recall.
- Confusion matrix saved as ``confusion_matrix.png``.
- ``sklearn.metrics.classification_report`` saved as ``classification_report.txt``.
- ``metrics.json`` with all numbers for programmatic use.
- ``comparison.json`` with delta vs previous model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate model on the test split and save all artefacts.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model (logits output, no softmax).
    test_ds : tf.data.Dataset
        Batched test dataset (images normalised to [0,1], integer labels).
    class_names : list[str]
        Ordered class names matching label indices.
    output_dir : Path
        Directory to save confusion_matrix.png, classification_report.txt,
        and metrics.json.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, loss, per_class (list),
        confusion_matrix (nested list).
    """
    # ── Keras evaluate for loss / accuracy / precision / recall ─────
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )

    loss, accuracy, precision, recall = model.evaluate(test_ds, verbose=1)

    logger.info(
        "Test — accuracy=%.4f, precision=%.4f, recall=%.4f, loss=%.4f",
        accuracy, precision, recall, loss,
    )

    # ── Collect predictions for confusion matrix ────────────────────
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in test_ds:
        logits = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(logits, axis=1).tolist())
        # labels are one-hot encoded from data.py
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())

    # ── Confusion matrix ────────────────────────────────────────────
    all_labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    _save_confusion_matrix(cm, class_names, output_dir)

    # ── Classification report ───────────────────────────────────────
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=all_labels,
        digits=4,
        zero_division=0,
    )
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report_str, encoding="utf-8")
    logger.info("Classification report saved to %s", report_path)
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(report_str)

    # ── Per-class stats from sklearn ────────────────────────────────
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        labels=all_labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    per_class = []
    for name in class_names:
        stats = report_dict.get(name, {})
        per_class.append({
            "class": name,
            "precision": round(stats.get("precision", 0), 4),
            "recall": round(stats.get("recall", 0), 4),
            "f1": round(stats.get("f1-score", 0), 4),
            "support": int(stats.get("support", 0)),
        })

    macro = report_dict.get("macro avg", {})
    macro_f1 = round(macro.get("f1-score", 0), 4)

    # ── Build metrics dict ──────────────────────────────────────────
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "loss": round(float(loss), 4),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    # Save metrics.json
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    logger.info("Metrics saved to %s", output_dir / "metrics.json")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Confusion matrix plot
# ═══════════════════════════════════════════════════════════════════════════

def _save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: Path,
) -> None:
    """Save confusion matrix as a PNG image (same style as Kaggle notebook)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)

    # Annotate cells
    threshold = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()

    path = output_dir / "confusion_matrix.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Comparison with previous model
# ═══════════════════════════════════════════════════════════════════════════

def compare_with_previous(
    new_metrics: Dict[str, Any],
    previous_metrics_path: Optional[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    """Compare new metrics against a previous model's metrics.json.

    Returns a comparison dict and saves it as comparison.json.
    """
    if previous_metrics_path is None or not previous_metrics_path.exists():
        comparison = {
            "has_previous": False,
            "recommendation": "promote",
            "reason": "No previous model to compare against.",
        }
    else:
        try:
            prev = json.loads(previous_metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            comparison = {
                "has_previous": False,
                "recommendation": "promote",
                "reason": "Could not read previous metrics.",
            }
        else:
            acc_delta = new_metrics["accuracy"] - prev.get("accuracy", 0)
            f1_delta = new_metrics["macro_f1"] - prev.get("macro_f1", 0)

            should_promote = acc_delta >= 0 or (f1_delta > 0.01 and acc_delta > -0.02)

            comparison = {
                "has_previous": True,
                "previous_accuracy": prev.get("accuracy"),
                "new_accuracy": new_metrics["accuracy"],
                "accuracy_delta": round(acc_delta, 4),
                "previous_f1": prev.get("macro_f1"),
                "new_f1": new_metrics["macro_f1"],
                "f1_delta": round(f1_delta, 4),
                "recommendation": "promote" if should_promote else "keep_previous",
                "reason": (
                    f"Accuracy {'improved' if acc_delta >= 0 else 'dropped'} by "
                    f"{abs(acc_delta):.4f}, F1 {'improved' if f1_delta >= 0 else 'dropped'} "
                    f"by {abs(f1_delta):.4f}."
                ),
            }

    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8",
    )
    logger.info("Comparison: %s", comparison.get("recommendation"))
    return comparison
