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

@csrf_exempt
@require_POST
def api_local_push_model(request, version_tag=None):
    """[MASTER ROLE] Push a specific model (or the active model) to the Cloud Broker."""
    if settings.SITE_ROLE != 'MASTER':
        return JsonResponse({"error": "This action is only available on MASTER nodes."}, status=403)
    
    try:
        # If no version_tag provided, find the active model
        if not version_tag:
            from classifier.models import TrainingRun
            active_run = TrainingRun.objects.filter(is_active_model=True, status='completed').first()
            if not active_run:
                return JsonResponse({"error": "No active trained model to push."}, status=404)
            version_tag = active_run.run_name

        # Instantiate the sync command and call its push_model_to_cloud directly
        from classifier.management.commands.sync_to_cloud import Command as SyncCommand
        from django.core.management.color import no_style
        
        cmd = SyncCommand()
        cmd.stdout = __import__('io').StringIO()  # Capture output
        cmd.style = no_style()                    # Avoid ANSI in captured output
        cmd.push_model_to_cloud(version_tag)
        
        output = cmd.stdout.getvalue()
        if 'Error' in output or 'failed' in output.lower():
            return JsonResponse({"status": "error", "message": output.strip()}, status=500)
        
        return JsonResponse({"status": "success", "message": f"Model {version_tag} pushed to cloud successfully."})
    except Exception as e:
        logger.exception(f"Failed to push model {version_tag}")
        return JsonResponse({"error": str(e)}, status=500)

import os
import shutil
from pathlib import Path
from classifier.models import AppSettings

@csrf_exempt
@require_POST
def api_local_fetch_model(request):
    """[EDGE ROLE] Trigger the sync_to_cloud command to fetch the latest model."""
    if settings.SITE_ROLE != 'EDGE':
        return JsonResponse({"error": "This action is only available on EDGE nodes."}, status=403)
    
    try:
        call_command('sync_to_cloud')
        return JsonResponse({"status": "success", "message": "Model fetch completed."})
    except Exception as e:
        logger.exception("Local fetch model failed")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_POST
def api_local_activate_model(request, version_tag):
    """[EDGE ROLE] Manually activate a downloaded model."""
    if settings.SITE_ROLE != 'EDGE':
        return JsonResponse({"error": "This action is only available on EDGE nodes."}, status=403)

    try:
        versions_dir = Path(settings.MODELS_ROOT) / 'versions' / version_tag
        source_model = versions_dir / f"{version_tag}.keras"
        
        if not source_model.exists():
            return JsonResponse({"error": f"Model {version_tag} not found locally."}, status=404)

        target_model = Path(settings.MODELS_ROOT) / 'logits_mdl.keras'
        
        # Copy the new model over the old one
        shutil.copy2(source_model, target_model)
        
        # Update AppSettings
        AppSettings.set('active_model_version', version_tag, 'The currently active model version.')
        
        # Dynamic hot-reload model in memory
        from classifier.model_loader import load_model
        load_model(str(target_model))
        logger.info("Edge hot-reload successful for model '%s'", version_tag)
        
        # Also deploy classes.txt if present
        source_classes = versions_dir / "classes.txt"
        target_classes = Path(settings.BASE_DIR) / "models" / "classes.txt"
        if source_classes.exists():
            shutil.copy2(str(source_classes), str(target_classes))
            from classifier.views.helpers import MODEL_CLASS_NAMES
            new_names = [l.strip() for l in open(target_classes) if l.strip()]
            MODEL_CLASS_NAMES.clear()
            MODEL_CLASS_NAMES.extend(new_names)
            logger.info("Edge classes list successfully reloaded: %s", MODEL_CLASS_NAMES)

        return JsonResponse({
            "status": "success", 
            "message": f"Model {version_tag} activated and hot-reloaded successfully!"
        })
    except Exception as e:
        logger.exception("Model activation failed")
        return JsonResponse({"error": str(e)}, status=500)

def api_local_list_models(request):
    """[EDGE ROLE] List locally downloaded models."""
    if settings.SITE_ROLE != 'EDGE':
        return JsonResponse({"error": "This action is only available on EDGE nodes."}, status=403)
    
    versions_dir = Path(settings.MODELS_ROOT) / 'versions'
    models_list = []
    
    active_version = AppSettings.get('active_model_version', 'v1')
    
    if versions_dir.exists():
        for d in versions_dir.iterdir():
            if d.is_dir():
                keras_file = d / f"{d.name}.keras"
                if keras_file.exists():
                    size = os.path.getsize(keras_file)
                    models_list.append({
                        "version_tag": d.name,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "is_active": (d.name == active_version)
                    })
    
    # Sort descending by version name
    models_list.sort(key=lambda x: x["version_tag"], reverse=True)
    
    return JsonResponse({"models": models_list, "active_version": active_version})
