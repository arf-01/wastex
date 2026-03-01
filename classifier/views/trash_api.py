"""
Trash counter APIs — aggregated counts and time-series history.
"""

from __future__ import annotations

from typing import Dict

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from classifier.models import TrashCounter

from .helpers import MODEL_CLASS_NAMES


@require_GET
def api_trash_counts(request):
    """Return the latest total count per waste class.

    Query params
    ------------
    from : ISO date — filter records from this date.
    to   : ISO date — filter records up to this date.
    """
    date_from = request.GET.get("from")
    date_to = request.GET.get("to")

    counts: Dict[str, int] = {}
    for cls in MODEL_CLASS_NAMES:
        qs = TrashCounter.objects.filter(class_name=cls).order_by("-recorded_at")
        if date_from:
            qs = qs.filter(recorded_at__gte=date_from)
        if date_to:
            qs = qs.filter(recorded_at__lte=date_to)
        latest = qs.first()
        counts[cls] = latest.total_count if latest else 0

    return JsonResponse({
        "counts": counts,
        "total": sum(counts.values()),
        "filters": {"from": date_from, "to": date_to},
    })


@require_GET
def api_trash_history(request):
    """Return time-series trash counter records.

    Query params
    ------------
    from  : ISO date — filter records from this date.
    to    : ISO date — filter records up to this date.
    class : str      — filter to a single waste class.
    """
    date_from = request.GET.get("from")
    date_to = request.GET.get("to")
    class_name = request.GET.get("class")

    qs = TrashCounter.objects.all()
    if class_name:
        qs = qs.filter(class_name=class_name)
    if date_from:
        qs = qs.filter(recorded_at__gte=date_from)
    if date_to:
        qs = qs.filter(recorded_at__lte=date_to)

    records = qs.order_by("recorded_at").values(
        "class_name", "total_count", "recorded_at",
    )
    history = [
        {
            "class_name": r["class_name"],
            "total_count": r["total_count"],
            "recorded_at": r["recorded_at"].isoformat(),
        }
        for r in records
    ]
    return JsonResponse({"history": history})
