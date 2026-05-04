"""
Local Sync APIs — allow triggering cloud relay actions from the UI.
These endpoints run on the LOCAL (Edge/Master) nodes, NOT on the Cloud role.
"""

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.management import call_command
import logging

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
def api_local_trigger_push(request):
    """[EDGE ROLE] Trigger the sync_to_cloud command for OOD images."""
    if settings.SITE_ROLE != 'EDGE':
        return JsonResponse({"error": "This action is only available on EDGE nodes."}, status=403)
    
    try:
        # We use call_command to reuse the existing logic in management commands
        call_command('sync_to_cloud')
        return JsonResponse({"status": "success", "message": "Edge sync completed. Check console for details."})
    except Exception as e:
        logger.exception("Local sync push failed")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_POST
def api_local_trigger_pull(request):
    """[MASTER ROLE] Trigger the sync_to_cloud command to fetch images."""
    if settings.SITE_ROLE != 'MASTER':
        return JsonResponse({"error": "This action is only available on MASTER nodes."}, status=403)
    
    try:
        call_command('sync_to_cloud')
        return JsonResponse({"status": "success", "message": "Master fetch completed. New images should appear in staging."})
    except Exception as e:
        logger.exception("Local sync pull failed")
        return JsonResponse({"error": str(e)}, status=500)
