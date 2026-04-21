"""
Simple Pi inference endpoint that reuses existing classify logic.
Uses DRF TokenAuthentication for secure edge device access.
"""

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from classifier.views.classification import classify
from classifier.models import Bin
import logging

logger = logging.getLogger(__name__)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def api_pi_inference(request):
    """
    Pi inference endpoint with token authentication.
    
    Usage:
        curl -X POST http://localhost:8000/api/pi/inference/ \
             -H "Authorization: Token YOUR_TOKEN" \
             -F "image=@photo.jpg"
    
    The Pi must send the token in the Authorization header:
        Authorization: Token <user_token>
    
    Delegates to the main classify logic to ensure consistent behavior.
    """
    # The request is already authenticated at this point by TokenAuthentication
    return classify(request)


@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def api_pi_health(request):
    """
    Health check endpoint - requires auth so we can track bin online status.
    Pi uses this to verify backend is reachable before attempting inference,
    and to heartbeat its online status.
    
    Usage:
        curl -X GET http://localhost:8000/api/pi/health/?source=bin_01 \
             -H "Authorization: Token YOUR_TOKEN"
    """
    bin_id = request.GET.get("source")
    if bin_id:
        try:
            bin_obj, _ = Bin.objects.get_or_create(
                user=request.user,
                bin_id=bin_id
            )
            bin_obj.save(update_fields=["last_active"])
        except Exception as e:
            logger.warning("Failed to resolve bin during health check: %s", e)

    return JsonResponse({
        "status": "ok",
        "message": "WasteX Backend is Online",
        "auth_required": True,
        "auth_type": "Token",
        "bin_tracked": bool(bin_id),
    })

