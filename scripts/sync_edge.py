import os
import sys
import logging
import requests
from pathlib import Path

# Setup Django environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wastex.settings")

import django
django.setup()

from django.conf import settings
from django.core.files.storage import default_storage
from classifier.models import Image, Bin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_sync():
    cloud_url = os.environ.get("CLOUD_API_URL")
    cloud_token = os.environ.get("CLOUD_API_TOKEN")

    if not cloud_url or not cloud_token:
        logger.error("CLOUD_API_URL and CLOUD_API_TOKEN must be set in your .env file.")
        return

    cloud_url = cloud_url.rstrip("/")
    headers = {"Authorization": f"Token {cloud_token}"}

    # 1. Sync OOD Images
    # Find all OOD images that have been reviewed but not synced
    images_to_sync = Image.objects.filter(
        top_prediction__isnull=True, 
        reviewed=True, 
        is_synced_to_cloud=False
    )

    if not images_to_sync.exists():
        logger.info("No new reviewed OOD images to sync.")
    else:
        logger.info(f"Found {images_to_sync.count()} images to sync to Cloud.")

        for img in images_to_sync:
            logger.info(f"Syncing image {img.filename}...")
            
            # Determine source (Bin ID or default)
            source_id = img.bin.bin_id if img.bin else "Edge_Default"

            try:
                with default_storage.open(img.image.name, 'rb') as f:
                    files = {'image': (img.filename, f, 'image/jpeg')}
                    data = {
                        'label': img.assigned_label or "Miscellaneous Trash",
                        'source': source_id
                    }

                    response = requests.post(
                        f"{cloud_url}/api/sync/ood/receive/",
                        headers=headers,
                        data=data,
                        files=files,
                        timeout=30
                    )

                if response.status_code == 200:
                    logger.info(f"Successfully synced {img.filename}")
                    # Delete local file to save SD card space
                    try:
                        default_storage.delete(img.image.name)
                    except Exception as e:
                        logger.warning(f"Could not delete local file {img.image.name}: {e}")
                    
                    # Delete the record from local DB since we don't need it on Edge anymore
                    img.delete()
                else:
                    logger.error(f"Failed to sync {img.filename}: {response.status_code} - {response.text}")

            except Exception as e:
                logger.error(f"Error syncing {img.filename}: {e}")

    # 2. Check for new models (omitted for now per user request, can be added later)
    logger.info("Edge sync complete.")

if __name__ == "__main__":
    run_sync()
