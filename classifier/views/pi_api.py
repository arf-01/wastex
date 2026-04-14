"""
Simple Pi inference endpoint that reuses existing classify logic.
No new database fields or complex logic - just process the image and return results.
"""

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from classifier.model_loader import get_logits
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def api_pi_inference(request):
    """
    Simple endpoint for Raspberry Pi to send an image and get inference results.
    Delegates to the main classify logic to ensure consistent behavior with the dashboard.
    """
    from classifier.views.classification import classify
    return classify(request)

@require_http_methods(["GET"])
def api_pi_health(request):
    """
    Health check endpoint for the Pi to verify connectivity before sending images.
    """
    return JsonResponse({"status": "ok", "message": "WasteX Backend is Online"})
