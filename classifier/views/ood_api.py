"""
OOD image APIs — list, review, and label out-of-distribution images.

Scoped to the logged-in edge user: only images captured by their bins
(or manually uploaded with no bin association) are returned.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from django.db.models import Q

from classifier.models import Image
from classifier.decorators import edge_required

from .helpers import parse_json_body, safe_positive_int

logger = logging.getLogger(__name__)


@login_required
@edge_required
@require_GET
def api_ood_images(request):
    """List OOD images belonging to this user's bins, with pagination.

    Only returns images captured by bins owned by the current user,
    plus any images with no bin (manual web uploads).

    Query params
    ------------
    from     : ISO date       — uploaded after this date.
    to       : ISO date       — uploaded before this date.
    reviewed : 'true'|'false' — filter by review status.
    bin_id   : str            — filter to a specific bin.
    page     : int            — page number (default 1).
    per_page : int            — items per page (default 20).
    """
    date_from = request.GET.get("from")
    date_to = request.GET.get("to")
    reviewed_filter = request.GET.get("reviewed")
    bin_id_filter = request.GET.get("bin_id")
    page = safe_positive_int(request.GET.get("page"), 1)
    per_page = safe_positive_int(request.GET.get("per_page"), 20)

    # Scope to current user: images from their bins OR bin-less manual uploads
    qs = Image.objects.filter(
        top_prediction__isnull=True,
        all_predictions__isnull=False,
    ).filter(
        Q(bin__user=request.user) | Q(bin__isnull=True)
    ).select_related('bin').order_by("-uploaded_at")

    if bin_id_filter:
        qs = qs.filter(bin__bin_id=bin_id_filter)
    if date_from:
        qs = qs.filter(uploaded_at__gte=date_from)
    if date_to:
        qs = qs.filter(uploaded_at__lte=date_to)
    if reviewed_filter == "true":
        qs = qs.filter(reviewed=True)
    elif reviewed_filter == "false":
        qs = qs.filter(reviewed=False)

    total = qs.count()
    unreviewed = (
        qs.filter(reviewed=False).count() if reviewed_filter is None else None
    )
    start = (page - 1) * per_page
    images = qs[start : start + per_page]

    results = []
    for img in images:
        preds = img.all_predictions
        energy = None
        logits = None

        if isinstance(preds, dict):
            energy = preds.get("energy")
            logits = preds.get("logits")
        elif isinstance(preds, list):
            logits = preds

        results.append({
            "id": img.id,
            "filename": img.filename,
            "image_url": img.image.url if img.image else None,
            "energy": energy,
            "logits": logits,
            "uploaded_at": img.uploaded_at.isoformat(),
            "width": img.width,
            "height": img.height,
            "reviewed": img.reviewed,
            "assigned_label": img.assigned_label,
            "added_to_dataset": img.added_to_dataset,
            "bin_id": img.bin.bin_id if img.bin else None,
        })

    resp: Dict[str, Any] = {
        "images": results,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if total > 0 else 0,
    }
    if unreviewed is not None:
        resp["unreviewed"] = unreviewed

    return JsonResponse(resp)


@login_required
@edge_required
@csrf_exempt
@require_POST
def api_review_image(request, image_id: int):
    """Mark an OOD image as reviewed (or un-reviewed).

    Request body (JSON):
        reviewed : bool (default True)
    """
    from django.db.models import Q
    try:
        img = Image.objects.get(
            Q(id=image_id),
            Q(bin__user=request.user) | Q(bin__isnull=True)
        )
    except Image.DoesNotExist:
        return JsonResponse({"error": "Image not found."}, status=404)

    body = parse_json_body(request)
    img.reviewed = body.get("reviewed", True)
    img.save(update_fields=["reviewed"])

    logger.info("Image %d review status → %s", image_id, img.reviewed)
    return JsonResponse({"id": img.id, "reviewed": img.reviewed})


@login_required
@edge_required
@csrf_exempt
@require_POST
def api_label_image(request, image_id: int):
    """Assign a class label to an OOD image.

    If the label is new, it will only exist on this Image record.

    Request body (JSON):
        label : str — the class label to assign (required, non-empty).
    """
    try:
        img = Image.objects.get(id=image_id)
    except Image.DoesNotExist:
        return JsonResponse({"error": "Image not found."}, status=404)

    body = parse_json_body(request)
    label = body.get("label", "").strip()

    if not label:
        return JsonResponse({"error": "Label is required."}, status=400)


    img.assigned_label = label
    img.reviewed = True
    img.save(update_fields=["assigned_label", "reviewed"])

    logger.info("Image %d labelled as '%s'", image_id, label)
    return JsonResponse({
        "id": img.id,
        "assigned_label": img.assigned_label,
        "reviewed": True,
    })
