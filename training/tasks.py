"""
Background thread launcher and status helpers for training runs.

Simple threading-based approach for single-user / development use.
For production, swap in Celery or Django-Q.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from classifier.models import TrainingRun
from .config import TrainingConfig
from .runner import run_training

logger = logging.getLogger(__name__)

# Module-level lock to prevent concurrent training runs
_training_lock = threading.Lock()


def start_training(config: TrainingConfig) -> Optional[str]:
    """Launch a training run in a background thread.

    Returns
    -------
    str | None
        The run_name if started successfully, or None if another
        run is already in progress.
    """
    if not _training_lock.acquire(blocking=False):
        logger.warning("Training already in progress — refusing to start.")
        return None

    version_name = config.resolve_dataset_version()

    def _run():
        try:
            run_training(config)
        except Exception:
            logger.exception("Background training failed")
        finally:
            _training_lock.release()

    thread = threading.Thread(target=_run, name="training-runner", daemon=True)
    thread.start()
    logger.info("Background training started for dataset '%s'", version_name)

    return version_name


def is_training_running() -> bool:
    """Return True if a training run is currently in progress."""
    return _training_lock.locked()


def get_latest_run() -> Optional[TrainingRun]:
    """Return the most recently created training run."""
    return TrainingRun.objects.order_by("-created_at").first()


def get_active_model_run() -> Optional[TrainingRun]:
    """Return the currently active (serving) training run."""
    return TrainingRun.objects.filter(is_active_model=True).first()
