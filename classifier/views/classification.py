"""
Image classification endpoint — accept an upload, run inference, return results.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from PIL import Image as PILImage

from classifier.model_loader import get_logits
from classifier.models import Image, TrashItem, Bin

from .helpers import ALLOWED_CONTENT_TYPES, MAX_UPLOAD_SIZE, MODEL_CLASS_NAMES

logger = logging.getLogger(__name__)


@csrf_exempt
@require_POST
def classify(request):
    """Accept an uploaded image, run inference, and return results.

    Workflow
    -------
    1. Validate the upload (presence, size, content-type).
    2. Save temporarily via Django storage.
    3. Run model inference to get logits, energy, OOD flag.
    4. OOD → persist ``Image`` record for operator review.
       In-distribution → increment ``TrashCounter``, delete file.
    5. Return JSON with logits, energy, OOD flag, predicted class.
    """
    if "image" not in request.FILES:
        return JsonResponse({"error": "No image file provided."}, status=400)

    image_file = request.FILES["image"]
    
    # Try to extract and resolve bin from 'source' payload if token auth is used
    bin_obj = None
    source = request.POST.get("source")
    if source and request.user.is_authenticated:
        try:
            bin_obj, _ = Bin.objects.get_or_create(
                user=request.user,
                bin_id=source
            )
            # update last_active
            bin_obj.save(update_fields=["last_active"])
        except Exception as e:
            logger.warning("Failed to resolve bin: %s", e)

    content_type = image_file.content_type
    if not content_type:
        import mimetypes
        content_type, _ = mimetypes.guess_type(image_file.name)
    
    if content_type not in ALLOWED_CONTENT_TYPES:
        return JsonResponse(
            {"error": f"Unsupported file type: {content_type}"},
            status=400,
        )

    if image_file.size > MAX_UPLOAD_SIZE:
        return JsonResponse(
            {"error": f"File too large ({image_file.size:,} bytes). "
                      f"Max {MAX_UPLOAD_SIZE:,}."},
            status=400,
        )

    import tempfile
    from django.core.files.storage import default_storage, FileSystemStorage

    try:
        # Create a temporary local storage to handle the file for inference
        # This prevents FileNotFoundError when DEFAULT_FILE_STORAGE is S3
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.name).suffix) as tmp:
            for chunk in image_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            # Perform inference on the local temp file
            logits, energy, ood = get_logits(tmp_path)
            
            # Save the file to permanent storage (S3 or Local)
            with open(tmp_path, 'rb') as f:
                file_path = default_storage.save(
                    f"uploads/{image_file.name}",
                    ContentFile(f.read()),
                )
            
            # Get the path for metadata if it's local storage
            if hasattr(default_storage, 'path'):
                full_path = Path(default_storage.path(file_path))
            else:
                full_path = Path(tmp_path) # Use temp file for metadata stats if S3
        finally:
            # We'll delete the temp file after we're done or if we fail
            if os.path.exists(tmp_path):
                # Only delete if we are ID (otherwise record creation needs stats)
                # But wait, we can get stats now
                tmp_stat = os.stat(tmp_path)
                pass 

    except Exception:
        logger.exception("Inference failed for file %s", image_file.name)
        return JsonResponse({"error": "Classification failed."}, status=500)
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            # If it was ID, we already deleted the saved one. 
            # We always delete tmp eventually.
            pass

    # Re-reading stats from tmp before it's gone
    file_size = tmp_stat.st_size
    from PIL import Image as PILImage
    with PILImage.open(tmp_path) as img:
        width, height = img.size
    
    os.unlink(tmp_path)

    saved = False
    predicted_class = None

    if ood:
        Image.objects.create(
            image=file_path,
            filename=image_file.name,
            file_size=file_size,
            width=width,
            height=height,
            all_predictions={
                "logits": logits.tolist(),
                "energy": float(energy),
            },
            uploaded_at=timezone.now(),
            classified_at=timezone.now(),
            bin=bin_obj,
        )
        saved = True
        logger.info("OOD image saved: %s (energy=%.4f)", image_file.name, energy)
    else:
        predicted_class_index = int(np.argmax(logits))
        predicted_class = MODEL_CLASS_NAMES[predicted_class_index]
        TrashItem.objects.create(class_name=predicted_class, bin=bin_obj)
        default_storage.delete(file_path)
        logger.info(
            "In-distribution: %s → %s (energy=%.4f)",
            image_file.name, predicted_class, energy,
        )

    return JsonResponse({
        "logits": logits.tolist(),
        "energy": float(energy),
        "ood": bool(ood),
        "predicted_class": predicted_class,
        "saved_to_db": saved,
    })
