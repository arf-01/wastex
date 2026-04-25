"""
Cloud Broker Sync APIs — handles communication between Edge, Master, and Cloud nodes.

Roles:
- CLOUD  : Receives OOD images, stores them in B2, registry for Master to download.
- EDGE   : Pushes labeled OOD images to Cloud. Polls for new models.
- MASTER : Pulls OOD images from Cloud for training. Pushes new models to Cloud.
"""

from __future__ import annotations

import logging
import os
import requests
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from classifier.models import Image, PendingImage, ModelRelease

logger = logging.getLogger(__name__)

# ── Cloud Role Endpoints ───────────────────────────────────────────────────

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_receive_ood(request):
    """[CLOUD ROLE] Accept an OOD image from an Edge site.
    
    Expects multipart/form-data:
        image  : file
        label  : str
        source : str (edge_site_id)
    """
    if settings.SITE_ROLE != 'CLOUD':
        return JsonResponse({"error": "This node is not configured as a CLOUD broker."}, status=403)

    if 'image' not in request.FILES:
        return JsonResponse({"error": "No image provided."}, status=400)

    image_file = request.FILES['image']
    label = request.data.get('label')
    source = request.data.get('source')

    if not label or not source:
        return JsonResponse({"error": "label and source are required."}, status=400)

    # Django-storages will handle the upload to B2 automatically because 
    # DEFAULT_FILE_STORAGE is set to S3Boto3Storage in settings.py for CLOUD role.
    try:
        # We create a dummy Image record just to trigger the storage backend
        # but the real "ledger" is the PendingImage model.
        img_obj = Image.objects.create(
            image=image_file,
            filename=image_file.name,
            assigned_label=label,
            top_prediction=None, # Mark as OOD
            reviewed=True,
        )

        # Create the ledger entry for the Master to find
        PendingImage.objects.create(
            edge_site_id=source,
            b2_file_key=img_obj.image.name,
            label=label,
            is_ready_for_master=True
        )

        return JsonResponse({
            "status": "success",
            "b2_key": img_obj.image.name
        })
    except Exception as e:
        logger.exception("Failed to receive OOD image on cloud")
        return JsonResponse({"error": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_get_pending_images(request):
    """[CLOUD ROLE] List all images ready for Master to download."""
    if settings.SITE_ROLE != 'CLOUD':
        return JsonResponse({"error": "Cloud role only."}, status=403)

    pending = PendingImage.objects.filter(is_ready_for_master=True)
    results = []
    for p in pending:
        results.append({
            "id": p.id,
            "edge_site_id": p.edge_site_id,
            "url": f"{settings.MEDIA_URL}{p.b2_file_key}",
            "b2_key": p.b2_file_key,
            "label": p.label,
            "created_at": p.created_at.isoformat()
        })
    
    return JsonResponse({"pending": results})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_cloud_release_model(request):
    """[CLOUD ROLE] Accept a new model from Master machine."""
    if settings.SITE_ROLE != 'CLOUD':
        return JsonResponse({"error": "Cloud role only."}, status=403)

    version_tag = request.data.get('version_tag')
    checksum = request.data.get('checksum')
    model_file = request.FILES.get('model')

    if not all([version_tag, checksum, model_file]):
        return JsonResponse({"error": "Missing fields."}, status=400)

    # Save model to B2
    # We store it in a specific folder "models/"
    release = ModelRelease.objects.create(
        version_tag=version_tag,
        checksum=checksum,
        notes=request.data.get('notes', ''),
        is_active=True
    )
    # The storage backend handles the upload
    # We might need a custom storage field or just manually save
    from django.core.files.storage import default_storage
    path = default_storage.save(f"releases/{version_tag}/{model_file.name}", model_file)
    release.b2_file_key = path
    release.save()

    # Deactivate older releases
    ModelRelease.objects.exclude(id=release.id).update(is_active=False)

    return JsonResponse({"status": "released", "version": version_tag})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_get_latest_release(request):
    """[CLOUD ROLE] Get the latest active model release info."""
    release = ModelRelease.objects.filter(is_active=True).first()
    if not release:
        return JsonResponse({"release": None})

    return JsonResponse({
        "version_tag": release.version_tag,
        "url": f"{settings.MEDIA_URL}{release.b2_file_key}",
        "checksum": release.checksum,
        "created_at": release.created_at.isoformat()
    })
