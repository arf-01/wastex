"""
Image classification endpoint — accept an upload, run inference, return results.
"""

from __future__ import annotations

import logging
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
from classifier.models import Image, TrashCounter

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

    if image_file.content_type not in ALLOWED_CONTENT_TYPES:
        return JsonResponse(
            {"error": f"Unsupported file type: {image_file.content_type}"},
            status=400,
        )

    if image_file.size > MAX_UPLOAD_SIZE:
        return JsonResponse(
            {"error": f"File too large ({image_file.size:,} bytes). "
                      f"Max {MAX_UPLOAD_SIZE:,}."},
            status=400,
        )

    try:
        file_path = default_storage.save(
            f"uploads/{image_file.name}",
            ContentFile(image_file.read()),
        )
        full_path = Path(default_storage.location) / file_path
        logits, energy, ood = get_logits(str(full_path))
    except Exception:
        logger.exception("Inference failed for file %s", image_file.name)
        return JsonResponse({"error": "Classification failed."}, status=500)

    saved = False
    predicted_class = None

    if ood:
        try:
            with PILImage.open(str(full_path)) as img:
                width, height = img.size
        except Exception:
            width, height = None, None

        Image.objects.create(
            image=file_path,
            filename=image_file.name,
            file_size=full_path.stat().st_size,
            width=width,
            height=height,
            all_predictions={
                "logits": logits.tolist(),
                "energy": float(energy),
            },
            uploaded_at=timezone.now(),
            classified_at=timezone.now(),
        )
        saved = True
        logger.info("OOD image saved: %s (energy=%.4f)", image_file.name, energy)
    else:
        predicted_class_index = int(np.argmax(logits))
        predicted_class = MODEL_CLASS_NAMES[predicted_class_index]
        TrashCounter.increment(predicted_class)
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
