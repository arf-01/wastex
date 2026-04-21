"""
Trash counter APIs — aggregated counts and time-series history.

All endpoints are scoped to the logged-in user's bins.
Unauthenticated requests are rejected with 401.
"""

from __future__ import annotations

from typing import Dict

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from classifier.models import TrashItem
from classifier.decorators import edge_required

from .helpers import MODEL_CLASS_NAMES


def _user_qs(request, qs):
    """Restrict a TrashItem queryset to the current user's bins only."""
    return qs.filter(bin__user=request.user)


@login_required
@edge_required
@require_GET
def api_trash_counts(request):
    """Return total count per waste class, scoped to the logged-in user's bins.

    Query params
    ------------
    from   : ISO datetime — filter records from this datetime.
    to     : ISO datetime — filter records up to this datetime.
    bin_id : str          — further narrow to a single bin (must belong to user).
    """
    date_from = request.GET.get("from")
    date_to   = request.GET.get("to")
    bin_id    = request.GET.get("bin_id")

    counts: Dict[str, int] = {}
    for cls in MODEL_CLASS_NAMES:
        qs = _user_qs(request, TrashItem.objects.filter(class_name=cls))
        if date_from:
            qs = qs.filter(recorded_at__gte=date_from)
        if date_to:
            qs = qs.filter(recorded_at__lte=date_to)
        if bin_id:
            qs = qs.filter(bin__bin_id=bin_id)
        counts[cls] = qs.count()

    return JsonResponse({
        "counts": counts,
        "total": sum(counts.values()),
        "filters": {"from": date_from, "to": date_to, "bin_id": bin_id},
    })


@login_required
@edge_required
@require_GET
def api_trash_history(request):
    """Return hourly time-series trash event counts, scoped to the logged-in user.

    Query params
    ------------
    from   : ISO datetime — filter records from this datetime.
    to     : ISO datetime — filter records up to this datetime.
    class  : str          — filter to a single waste class.
    bin_id : str          — filter to a specific bin (must belong to user).
    """
    from django.db.models import Count
    from django.db.models.functions import TruncHour

    date_from  = request.GET.get("from")
    date_to    = request.GET.get("to")
    class_name = request.GET.get("class")
    bin_id     = request.GET.get("bin_id")

    # Always start scoped to this user's bins
    qs = _user_qs(request, TrashItem.objects.all())

    if class_name:
        qs = qs.filter(class_name=class_name)
    if date_from:
        qs = qs.filter(recorded_at__gte=date_from)
    if date_to:
        qs = qs.filter(recorded_at__lte=date_to)
    if bin_id:
        qs = qs.filter(bin__bin_id=bin_id)

    records = (
        qs.annotate(hour=TruncHour("recorded_at"))
          .values("class_name", "hour")
          .annotate(count=Count("id"))
          .order_by("hour")
    )

    # Build running totals — seed from everything before the date_from window
    running_totals = {cls: 0 for cls in MODEL_CLASS_NAMES}
    if date_from:
        base_qs = _user_qs(request, TrashItem.objects.filter(recorded_at__lt=date_from))
        if bin_id:
            base_qs = base_qs.filter(bin__bin_id=bin_id)
        for cls in MODEL_CLASS_NAMES:
            running_totals[cls] = base_qs.filter(class_name=cls).count()

    history = []
    for r in records:
        cls = r["class_name"]
        running_totals[cls] += r["count"]
        history.append({
            "class_name": cls,
            "total_count": running_totals[cls],
            "recorded_at": r["hour"].isoformat(),
        })

    return JsonResponse({"history": history})
