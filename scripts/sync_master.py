import os
import sys
import logging
import requests
from pathlib import Path
from datetime import datetime

# Setup Django environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wastex.settings")

import django
django.setup()

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from classifier.models import Image

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

    # 1. Fetch pending images from Cloud Broker
    logger.info("Fetching pending images from Cloud Broker...")
    try:
        response = requests.get(f"{cloud_url}/api/sync/ood/pending/", headers=headers, timeout=30)
        response.raise_for_status()
        pending_data = response.json().get("pending", [])
    except Exception as e:
        logger.error(f"Failed to fetch pending images: {e}")
        return

    if not pending_data:
        logger.info("No new pending images on the Cloud Broker.")
        return

    logger.info(f"Found {len(pending_data)} images to download.")
    downloaded_ids = []

    for item in pending_data:
        img_id = item.get("id")
        img_url = item.get("url")
        b2_key = item.get("b2_key")
        label = item.get("label")

        basename = Path(b2_key).name
        logger.info(f"Downloading {basename}...")

        try:
            # Download the image file
            img_resp = requests.get(img_url, timeout=60)
            img_resp.raise_for_status()

            # Save locally
            now = datetime.now()
            local_path = f"uploads/{now.year}/{now.month:02d}/{now.day:02d}/cloud_sync_{basename}"
            saved_path = default_storage.save(local_path, ContentFile(img_resp.content))

            # Insert into local database
            Image.objects.create(
                image=saved_path,
                filename=basename,
                assigned_label=label,
                top_prediction=None, # OOD marker
                reviewed=True,       # Already reviewed by Edge Operator
                is_synced_to_cloud=True # We just pulled it from the cloud
            )

            downloaded_ids.append(img_id)
            logger.info(f"Successfully saved and registered {basename}")

        except Exception as e:
            logger.error(f"Failed to download and process {basename}: {e}")

    # 2. Acknowledge downloaded images
    if downloaded_ids:
        logger.info(f"Acknowledging {len(downloaded_ids)} images with Cloud Broker...")
        try:
            ack_resp = requests.post(
                f"{cloud_url}/api/sync/ood/downloaded/",
                headers=headers,
                json={"image_ids": downloaded_ids},
                timeout=30
            )
            ack_resp.raise_for_status()
            logger.info("Acknowledgement successful.")
        except Exception as e:
            logger.error(f"Failed to acknowledge downloaded images: {e}")

    logger.info("Master sync complete.")

if __name__ == "__main__":
    run_sync()
