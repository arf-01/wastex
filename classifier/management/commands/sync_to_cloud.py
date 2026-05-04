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
from pathlib import Path
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
            if options.get('fetch_model'):
                self.fetch_model_from_cloud()
            else:
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
            synced_successfully = False
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
                        synced_successfully = True
                    else:
                        self.stdout.write(self.style.ERROR(f"  [!] Failed {img.filename}: {resp.text}"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  [!] Error syncing {img.filename}: {str(e)}"))

            # CLEANUP: Move outside the 'with open' block so Windows releases the lock
            if synced_successfully:
                try:
                    if os.path.exists(img.image.path):
                        os.remove(img.image.path)
                        self.stdout.write(self.style.WARNING(f"      [x] Purged local file: {img.filename}"))
                    
                    # Also delete from the Edge database so the UI doesn't show 404s
                    img.delete()
                    self.stdout.write(self.style.WARNING(f"      [x] Purged database record: {img.filename}"))
                except Exception as cleanup_err:
                    self.stdout.write(self.style.ERROR(f"      [!] Cleanup failed: {str(cleanup_err)}"))

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
                    class_dir = staging_dir / p['label']
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    target_path = class_dir / p['b2_key'].split('/')[-1]
                    
                    # 1. Download if missing
                    if not target_path.exists():
                        self.stdout.write(f"  [-] Downloading {p['url']} via B2 Authenticated Client...")
                        try:
                            import boto3
                            from botocore.client import Config

                            s3_client = boto3.client(
                                's3',
                                endpoint_url=settings.AWS_S3_ENDPOINT_URL,
                                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                config=Config(signature_version='s3v4'),
                                region_name=settings.AWS_S3_REGION_NAME
                            )

                            s3_client.download_file(
                                settings.AWS_STORAGE_BUCKET_NAME,
                                p['b2_key'],
                                str(target_path)
                            )
                            self.stdout.write(self.style.SUCCESS(f"  [+] Saved to {target_path}"))
                        except Exception as download_err:
                            self.stdout.write(self.style.ERROR(f"  [!] B2 Download failed: {str(download_err)}"))
                            continue
                    else:
                        self.stdout.write(f"  [=] File already exists on disk: {target_path.name}")

                    # 2. Always register in Master's local database
                    # This makes the image appear in the "Staged" count
                    _, created = Image.objects.get_or_create(
                        filename=target_path.name,
                        defaults={
                            'image': f"staging/{p['label']}/{target_path.name}",
                            'assigned_label': p['label'],
                            'is_synced_to_cloud': True,
                            'added_to_dataset': False
                        }
                    )
                    if created:
                        self.stdout.write(f"  [+] Registered in database.")

                    # 3. Notify Cloud to clear the queue
                    try:
                        requests.post(
                            f"{broker_url.rstrip('/')}/api/sync/ood/downloaded/",
                            headers=headers,
                            json={'image_ids': [p['id']]},
                            timeout=10
                        )
                    except:
                        pass
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

    def fetch_model_from_cloud(self):
        """[EDGE ROLE] Fetch the latest active model from the cloud."""
        broker_url = os.getenv('CLOUD_BROKER_URL')
        token = os.getenv('CLOUD_BROKER_TOKEN')

        if not broker_url or not token:
            self.stdout.write(self.style.ERROR("CLOUD_BROKER_URL or CLOUD_BROKER_TOKEN not set."))
            return

        headers = {'Authorization': f'Token {token}'}
        self.stdout.write("Checking for latest model release...")

        try:
            resp = requests.get(
                f"{broker_url.rstrip('/')}/api/sync/model/latest/",
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if not data.get('release') and 'version_tag' not in data:
                    self.stdout.write("No active models found on Cloud Broker.")
                    return
                
                version_tag = data.get('version_tag')
                url = data.get('url')
                checksum = data.get('checksum')

                # Create versions directory if not exists
                versions_dir = Path(settings.MODELS_ROOT) / 'versions' / version_tag
                versions_dir.mkdir(parents=True, exist_ok=True)
                
                target_path = versions_dir / f"{version_tag}.keras"

                if target_path.exists():
                    self.stdout.write(self.style.SUCCESS(f"  [=] Model {version_tag} is already downloaded."))
                    return

                self.stdout.write(f"  [-] Downloading {version_tag} from {url} ...")
                
                # Stream the download since it's a large file
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(target_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                self.stdout.write("  [-] Verifying checksum...")
                import hashlib
                sha256 = hashlib.sha256()
                with open(target_path, "rb") as f:
                    sha256.update(f.read())
                downloaded_checksum = sha256.hexdigest()

                if downloaded_checksum != checksum:
                    self.stdout.write(self.style.ERROR("  [!] Checksum mismatch! Corrupted download."))
                    os.remove(target_path)
                    return
                
                self.stdout.write(self.style.SUCCESS(f"  [+] Model {version_tag} downloaded and verified successfully!"))
            else:
                self.stdout.write(self.style.ERROR(f"  [!] Cloud error: {resp.text}"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  [!] Connection error: {str(e)}"))

    def add_arguments(self, parser):
        parser.add_argument(
            "--release",
            type=str,
            help="Specify a model version tag to release to the cloud (Master role only).",
        )
        parser.add_argument(
            "--fetch-model",
            action="store_true",
            help="Fetch the latest model from the cloud (Edge role only).",
        )
