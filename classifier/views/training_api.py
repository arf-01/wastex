"""
Training API endpoints.

POST /api/training/start/    – Kick off a new training run (background).
GET  /api/training/status/   – Latest run status (+ optional run_name filter).
POST /api/training/promote/  – Promote a completed run to active model.
"""

from __future__ import annotations

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from classifier.models import TrainingRun
from training.config import TrainingConfig
from training.tasks import (
    get_active_model_run,
    get_latest_run,
    is_training_running,
    start_training,
)

logger = logging.getLogger(__name__)


@csrf_exempt
@require_POST
def api_training_start(request):
    """Start a new training run.

    Accepts an optional JSON body to override default config values.
    Returns 409 if a run is already in progress.
    """
    if is_training_running():
        return JsonResponse(
            {"error": "A training run is already in progress."},
            status=409,
        )

    # Build config – merge request overrides with defaults
    overrides = {}
    if request.body:
        try:
            overrides = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON body."}, status=400)

    try:
        config = TrainingConfig(**overrides)
    except TypeError as exc:
        return JsonResponse({"error": f"Bad config: {exc}"}, status=400)

    version_name = start_training(config)
    if version_name is None:
        return JsonResponse(
            {"error": "Could not start training (lock contention)."},
            status=409,
        )

    return JsonResponse({
        "status": "started",
        "dataset_version": version_name,
        "config": config.to_dict(),
    })


@require_GET
def api_training_status(request):
    """Return the latest training run status.

    Query params
    ------------
    run_name : str, optional
        Filter to a specific run name.
    """
    run_name = request.GET.get("run_name")
    if run_name:
        run = TrainingRun.objects.filter(run_name=run_name).first()
    else:
        run = get_latest_run()

    if run is None:
        return JsonResponse({"run": None, "training_running": is_training_running()})

    return JsonResponse({
        "training_running": is_training_running(),
        "run": {
            "run_name": run.run_name,
            "dataset_version_name": run.dataset_version_name,
            "status": run.status,
            "error_message": run.error_message or "",
            "epochs_completed": run.epochs_completed,
            "test_accuracy": run.test_accuracy,
            "test_f1": run.test_f1,
            "is_active_model": run.is_active_model,
            "model_path": run.model_path or None,
            "config": run.config or {},
            "metrics_summary": run.metrics_summary or {},
            "comparison": run.comparison or {},
            "created_at": run.created_at.isoformat(),
        },
    })


@require_GET
def api_training_history(request):
    """Return all training runs, newest first.

    Query params
    ------------
    limit : int, optional
        Max number of runs to return (default: all).
    """
    limit = request.GET.get("limit")
    qs = TrainingRun.objects.all().order_by("-created_at")
    if limit:
        try:
            qs = qs[: int(limit)]
        except (ValueError, TypeError):
            pass

    runs = []
    for run in qs:
        duration = None
        if run.started_at and run.finished_at:
            delta = run.finished_at - run.started_at
            total_secs = int(delta.total_seconds())
            hours, rem = divmod(total_secs, 3600)
            mins, secs = divmod(rem, 60)
            duration = f"{hours}h {mins}m {secs}s"

        runs.append({
            "run_name": run.run_name,
            "dataset_version_name": run.dataset_version_name,
            "status": run.status,
            "epochs_completed": run.epochs_completed,
            "test_accuracy": run.test_accuracy,
            "test_f1": run.test_f1,
            "is_active_model": run.is_active_model,
            "model_path": run.model_path or None,
            "error_message": run.error_message or "",
            "num_classes": run.num_classes,
            "num_train_samples": run.num_train_samples,
            "num_val_samples": run.num_val_samples,
            "num_test_samples": run.num_test_samples,
            "metrics_summary": run.metrics_summary or {},
            "comparison": run.comparison or {},
            "config": run.config or {},
            "duration": duration,
            "created_at": run.created_at.isoformat(),
            "updated_at": run.updated_at.isoformat() if hasattr(run, 'updated_at') and run.updated_at else None,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        })

    return JsonResponse({
        "training_running": is_training_running(),
        "count": len(runs),
        "runs": runs,
    })


@csrf_exempt
@require_POST
def api_training_promote(request):
    """Promote a completed run to the active serving model.

    Expects JSON: {"run_name": "..."}.
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    run_name = body.get("run_name")
    if not run_name:
        return JsonResponse({"error": "run_name is required."}, status=400)

    run = TrainingRun.objects.filter(run_name=run_name).first()
    if run is None:
        return JsonResponse({"error": "Run not found."}, status=404)

    if run.status != "completed":
        return JsonResponse(
            {"error": f"Cannot promote a run with status '{run.status}'."},
            status=400,
        )

    # Deactivate all others, activate this one
    TrainingRun.objects.filter(is_active_model=True).update(is_active_model=False)
    run.is_active_model = True
    run.save(update_fields=["is_active_model"])

    # Reload the model into memory
    try:
        from classifier.model_loader import load_model
        load_model(run.model_path)
        logger.info("Promoted and loaded model from run '%s'", run_name)
    except Exception:
        logger.exception("Model promoted in DB but hot-reload failed")

    return JsonResponse({
        "status": "promoted",
        "run_name": run.run_name,
        "model_path": run.model_path,
    })
