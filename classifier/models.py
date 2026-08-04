"""
Database models for the WasteX classification system.

Models
------
DatasetVersion  – Versioned snapshots of the training dataset (v1, v2, …).
VersionEntry    – Delta-based membership: links a physical file to a version.
DatasetClass    – Canonical, growing registry of waste class labels.
Image           – Uploaded images with classification / OOD results.
TrashCounter    – Aggregated per-class item counts (time-series).

Delta-based versioning
----------------------
Instead of duplicating every file on disk when a new version is created,
we store **one row per image per version** in the ``VersionEntry`` table.
Each row records the physical file path, the split (train/test/val), and
the class label.  Creating a new version simply *inherits* the parent's
entries (copies lightweight DB rows) and *adds* new entries for staged
OOD images — no ``shutil.copytree``, no disk duplication.

To resolve "all images in version X", query ``VersionEntry`` for that
version.  The actual bytes live either in ``datasets/v1/…`` (the seed
dataset) or in ``media/uploads/…`` (operator-labelled OOD images).
"""

from django.core.validators import MinLengthValidator
from django.db import models
from django.db.models import F
from django.utils import timezone
from django.conf import settings



# ── Uploaded images ─────────────────────────────────────────────────────────

class Image(models.Model):
    """An uploaded image together with its classification / OOD metadata.

    In-distribution images are counted via :class:`TrashCounter` and the
    file is deleted.  OOD images are persisted here so an operator can
    inspect, label, and eventually add them to a new dataset version.

    Key fields for operator workflows:
        reviewed        – The operator has seen this image on the inspect page.
        assigned_label  – The label the operator chose (may be a new class).
        added_to_dataset – The image has been copied into a dataset version.
    """

    # ── File ────────────────────────────────────────────────────────────
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    filename = models.CharField(max_length=255, help_text='Original upload filename.')
    file_size = models.PositiveIntegerField(null=True, blank=True, help_text='Size in bytes.')
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)

    # ── Classification results ──────────────────────────────────────────
    top_prediction = models.CharField(
        max_length=100, null=True, blank=True, db_index=True,
        help_text='Predicted class (NULL for OOD images).',
    )
    confidence = models.FloatField(null=True, blank=True)
    all_predictions = models.JSONField(
        null=True, blank=True,
        help_text='Raw inference output: {"logits": [...], "energy": float}.',
    )

    # ── Operator review / labelling ─────────────────────────────────────
    reviewed = models.BooleanField(default=False, db_index=True)
    assigned_label = models.CharField(
        max_length=100, null=True, blank=True, db_index=True,
        help_text='Operator-assigned class label.',
    )


    
    # ── Cloud Sync ──────────────────────────────────────────────────────
    is_synced_to_cloud = models.BooleanField(
        default=False, 
        db_index=True,
        help_text='Whether this labeled OOD image has been pushed to the cloud broker.'
    )
    
    # ── Edge Tracking ───────────────────────────────────────────────────
    bin = models.ForeignKey(
        'Bin',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='ood_images',
        help_text='The edge bin that captured this OOD image.'
    )

    # ── Timestamps ──────────────────────────────────────────────────────
    uploaded_at = models.DateTimeField(default=timezone.now, db_index=True)
    classified_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'images'
        ordering = ['-uploaded_at']
        verbose_name = 'Image'
        verbose_name_plural = 'Images'
        indexes = [
            # Composite index for the inspect page (OOD images, paged by date)
            models.Index(
                fields=['top_prediction', 'reviewed', '-uploaded_at'],
                name='idx_image_ood_review',
            ),
        ]

    def __str__(self) -> str:
        return f"{self.filename} – {self.top_prediction or 'OOD'}"

    def save(self, *args, **kwargs):
        """Auto-populate filename from the uploaded image field if blank."""
        if not self.filename and self.image:
            self.filename = self.image.name
        super().save(*args, **kwargs)


# ── Trash counters (time-series) ────────────────────────────────────────────

class Bin(models.Model):
    """A physical edge capture station (e.g. Raspberry Pi) belonging to a user.
    
    Tracks the health/activity of the bin and anchors trash detections to it.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='bins',
        help_text='The edge user this bin belongs to.'
    )
    bin_id = models.CharField(
        max_length=100,
        db_index=True,
        help_text='Unique identifier string sent by the edge device.'
    )
    last_active = models.DateTimeField(
        auto_now=True,
        help_text='Last time this bin communicated with the server.'
    )
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'bins'
        unique_together = ('user', 'bin_id')
        verbose_name = 'Bin'
        verbose_name_plural = 'Bins'
        ordering = ['-last_active']

    def __str__(self) -> str:
        return f"{self.user.username} - {self.bin_id}"


class TrashItem(models.Model):
    """An individual event for a detected piece of trash.
    
    By storing one row per item, we avoid race conditions associated with 
    concurrently updating an integer counter, and we gain rich historical
    data filtered by bin.
    """
    bin = models.ForeignKey(
        Bin,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='trash_items',
        help_text='The bin that detected this item (null for manual uploads).'
    )
    class_name = models.CharField(max_length=100, db_index=True)
    recorded_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = 'trash_items'
        ordering = ['-recorded_at']
        verbose_name = 'Trash item event'
        verbose_name_plural = 'Trash item events'
        indexes = [
            models.Index(fields=['class_name', 'recorded_at']),
        ]

    def __str__(self) -> str:
        bin_str = self.bin.bin_id if self.bin else "Manual"
        return f"{self.class_name} detected by {bin_str} at {self.recorded_at:%Y-%m-%d %H:%M}"



# ── Application Settings (Installation Configuration) ──────────────────────

class AppSettings(models.Model):
    """Store configurable application settings set during installation.

    Used to store user-selected paths for:
    - Media root (where uploaded images are stored)
    - Datasets root (where training datasets are organized)
    - Models root (where trained models are saved)

    These are set once during installation and are read-only thereafter.

    Attributes:
        key:         Setting name (e.g., 'media_root', 'datasets_root')
        value:       Setting value (e.g., 'D:/WasteData/media')
        description: Human-readable description
        created_at:  When setting was created
        updated_at:  When setting was last modified
    """

    key = models.CharField(
        max_length=100,
        unique=True,
        help_text="Setting key (e.g., 'media_root')"
    )
    value = models.TextField(
        help_text="Setting value (e.g., '/path/to/folder')"
    )
    description = models.TextField(
        blank=True,
        help_text="Description of what this setting controls"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "App Setting"
        verbose_name_plural = "App Settings"

    def __str__(self):
        return f"{self.key} = {self.value}"

    @classmethod
    def get(cls, key: str, default=None):
        """Get a setting value, return default if not found.

        Args:
            key: Setting name
            default: Value to return if not found

        Returns:
            Setting value or default
        """
        try:
            setting = cls.objects.get(key=key)
            return setting.value
        except cls.DoesNotExist:
            return default

    @classmethod
    def set(cls, key: str, value: str, description: str = ""):
        """Set or update a setting value.

        Args:
            key: Setting name
            value: New value
            description: Optional description

        Returns:
            AppSettings instance
        """
        obj, created = cls.objects.update_or_create(
            key=key,
            defaults={'value': value, 'description': description}
        )
        return obj

# ── Cloud Broker Ledger (Used on Cloud Role) ────────────────────────────────

class PendingImage(models.Model):
    """The 'Waiting Room' for OOD images in the cloud.
    
    Used by the Cloud Broker to track images pushed from Edge sites that 
    are waiting to be downloaded by the Master machine.
    """
    edge_site_id = models.CharField(
        max_length=100, 
        db_index=True,
        help_text="The source site ID (e.g., NY_Facility_1)"
    )
    b2_file_key = models.CharField(
        max_length=500,
        help_text="Path in Backblaze B2 (e.g., incoming/ny1/img_99.jpg)"
    )
    label = models.CharField(
        max_length=100,
        help_text="The operator-assigned label from the Edge site."
    )
    is_ready_for_master = models.BooleanField(
        default=False,
        db_index=True,
        help_text="True once the file upload to B2 is confirmed."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'cloud_pending_images'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.edge_site_id} - {self.label} ({self.b2_file_key})"


class ModelRelease(models.Model):
    """Registry of trained models released by the Master machine.
    
    Used by Edge sites to poll for and download the latest .keras models.
    """
    version_tag = models.CharField(
        max_length=50,
        unique=True,
        help_text="e.g., v2.1"
    )
    b2_file_key = models.CharField(
        max_length=500,
        help_text="Path in Backblaze B2 (e.g., models/v2.1.keras)"
    )
    checksum = models.CharField(
        max_length=128,
        help_text="SHA256 checksum to verify download integrity."
    )
    notes = models.TextField(blank=True)
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text="Whether this is the recommended version for Edge sites."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'cloud_model_releases'
        ordering = ['-created_at']

    def __str__(self):
        return f"Release {self.version_tag} ({'Active' if self.is_active else 'Inactive'})"
