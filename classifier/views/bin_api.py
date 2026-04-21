"""
Bin API endpoints — returns per-bin health and statistics.
"""

from django.db.models import Count, Q
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.contrib.auth.decorators import login_required
from django.utils import timezone

from classifier.models import Bin, TrashItem, Image
from classifier.decorators import edge_required


@login_required
@edge_required
@require_GET
def api_bins(request):
    """Return a list of bins belonging to the current user.

    Each bin entry includes:
    - bin_id, last_active, is_online
    - trash_count  : total in-distribution trash items detected
    - ood_count    : total OOD images captured
    - ood_pending  : OOD images not yet reviewed / labelled
    """
    bins = Bin.objects.filter(user=request.user).order_by('-last_active')

    now = timezone.now()
    bin_list = []

    for b in bins:
        is_online = (now - b.last_active).total_seconds() < 900

        trash_count = TrashItem.objects.filter(bin=b).count()

        ood_qs = Image.objects.filter(
            bin=b,
            top_prediction__isnull=True,
            all_predictions__isnull=False,
        )
        ood_count = ood_qs.count()
        ood_pending = ood_qs.filter(reviewed=False).count()

        bin_list.append({
            "id": b.id,
            "bin_id": b.bin_id,
            "last_active": b.last_active.isoformat(),
            "is_online": is_online,
            "trash_count": trash_count,
            "ood_count": ood_count,
            "ood_pending": ood_pending,
        })

    return JsonResponse({"bins": bin_list})

