"""
Management command to sync local data to the Cloud Broker.

Usage:
    python manage.py sync_to_cloud

Behaviour:
- [EDGE Role]   : Pushes labeled OOD images and TrashItem counts to the cloud.
- [MASTER Role] : Pulls new training data from cloud, pushes new models.
"""

import os
import requests
import logging
from django.core.management.base import BaseCommand
from django.conf import settings
from classifier.models import Image, TrashItem

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Sync local data to/from the Cloud Broker."

    def handle(self, *args, **options):
        role = settings.SITE_ROLE
        self.stdout.write(f"Syncing as {role}...")

        if role == 'EDGE':
            self.sync_edge_to_cloud()
        elif role == 'MASTER':
            self.sync_master_to_cloud()
            # Also check for latest model releases to push
            if options.get('release'):
                self.push_model_to_cloud(options['release'])
        elif role == 'CLOUD':
            self.stdout.write(self.style.WARNING("Cloud nodes do not initiate sync. They wait for Edge/Master."))
        
    def sync_edge_to_cloud(self):
        """Find unsynced labeled images and push them."""
        broker_url = os.getenv('CLOUD_BROKER_URL')
        token = os.getenv('CLOUD_BROKER_TOKEN')
        site_id = os.getenv('SITE_ID', 'unknown_site')

        if not broker_url or not token:
            self.stdout.write(self.style.ERROR("CLOUD_BROKER_URL or CLOUD_BROKER_TOKEN not set."))
            return

        # 1. Sync OOD Images
        images_to_sync = Image.objects.filter(
            is_synced_to_cloud=False,
            assigned_label__isnull=False
        )

        self.stdout.write(f"Found {images_to_sync.count()} images to sync.")
        
        headers = {'Authorization': f'Token {token}'}
        
        for img in images_to_sync:
            try:
                # Open the physical file
                with open(img.image.path, 'rb') as f:
                    files = {'image': (img.filename, f, 'image/jpeg')}
                    data = {
                        'label': img.assigned_label,
                        'source': site_id
                    }
                    
                    resp = requests.post(
                        f"{broker_url.rstrip('/')}/api/sync/ood/receive/",
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=30
                    )
                    
                    if resp.status_code == 200:
                        img.is_synced_to_cloud = True
                        img.save(update_fields=['is_synced_to_cloud'])
                        self.stdout.write(self.style.SUCCESS(f"  [+] Synced: {img.filename}"))
                    else:
                        self.stdout.write(self.style.ERROR(f"  [!] Failed {img.filename}: {resp.text}"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  [!] Error syncing {img.filename}: {str(e)}"))

    def sync_master_to_cloud(self):
        """Pull new training data from cloud."""
        broker_url = os.getenv('CLOUD_BROKER_URL')
        token = os.getenv('CLOUD_BROKER_TOKEN')

        if not broker_url or not token:
            self.stdout.write(self.style.ERROR("CLOUD_BROKER_URL or CLOUD_BROKER_TOKEN not set."))
            return

        headers = {'Authorization': f'Token {token}'}
        
        # Pull pending images
        try:
            resp = requests.get(
                f"{broker_url.rstrip('/')}/api/sync/ood/pending/",
                headers=headers,
                timeout=30
            )
            if resp.status_code == 200:
                pending = resp.json().get('pending', [])
                self.stdout.write(f"Found {len(pending)} pending images on cloud.")
                
                # Staging area for Master
                staging_dir = Path(settings.DATASETS_ROOT) / 'staging'
                staging_dir.mkdir(parents=True, exist_ok=True)

                for p in pending:
                    # Download the file
                    class_dir = staging_dir / p['label']
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    target_path = class_dir / p['b2_key'].split('/')[-1]
                    
                    if target_path.exists():
                        self.stdout.write(f"  [=] Already exists: {target_path.name}")
                        continue

                    self.stdout.write(f"  [-] Downloading {p['url']} ...")
                    img_resp = requests.get(p['url'], timeout=60)
                    if img_resp.status_code == 200:
                        with open(target_path, 'wb') as f:
                            f.write(img_resp.content)
                        self.stdout.write(self.style.SUCCESS(f"  [+] Saved to {target_path}"))
                        
                        # Note: In a real flow, we would then tell the cloud
                        # to delete the image from B2 to keep the 10GB free.
                    else:
                        self.stdout.write(self.style.ERROR(f"  [!] Download failed: {img_resp.status_code}"))
            else:
                self.stdout.write(self.style.ERROR(f"  [!] Cloud error: {resp.text}"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  [!] Connection error: {str(e)}"))

    def push_model_to_cloud(self, version_tag):
        """Push a local .keras model to the cloud broker."""
        broker_url = os.getenv('CLOUD_BROKER_URL')
        token = os.getenv('CLOUD_BROKER_TOKEN')

        model_path = Path(settings.MODELS_ROOT) / 'versions' / version_tag / f"{version_tag}.keras"
        if not model_path.exists():
            self.stdout.write(self.style.ERROR(f"Model file not found: {model_path}"))
            return

        import hashlib
        sha256 = hashlib.sha256()
        with open(model_path, "rb") as f:
            sha256.update(f.read())
        checksum = sha256.hexdigest()

        self.stdout.write(f"Pushing model {version_tag} to cloud...")
        
        headers = {'Authorization': f'Token {token}'}
        with open(model_path, 'rb') as f:
            files = {'model': f}
            data = {
                'version_tag': version_tag,
                'checksum': checksum,
                'notes': f"Auto-released from Master machine"
            }
            resp = requests.post(
                f"{broker_url.rstrip('/')}/api/sync/model/release/",
                headers=headers,
                files=files,
                data=data,
                timeout=300 # Models are large
            )
            
            if resp.status_code == 200:
                self.stdout.write(self.style.SUCCESS(f"  [+] Model {version_tag} released successfully."))
            else:
                self.stdout.write(self.style.ERROR(f"  [!] Release failed: {resp.text}"))

    def add_arguments(self, parser):
        parser.add_argument(
            "--release",
            type=str,
            help="Specify a model version tag to release to the cloud (Master role only).",
        )
