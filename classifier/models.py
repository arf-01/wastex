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


# ── Dataset versioning ──────────────────────────────────────────────────────

class DatasetVersion(models.Model):
    """A versioned snapshot of the training / evaluation dataset.

    Versions follow a parent → child lineage.  The **actual images are
    never copied**; instead, each version's contents are defined by its
    ``VersionEntry`` rows.  Cached statistics (``total_images``,
    ``class_counts``, ``splits``) are denormalised for fast listing.

    Attributes:
        name:         Unique human-readable identifier, e.g. ``"v1"``.
        parent:       Optional predecessor version this was forked from.
        notes:        Free-text description / changelog.
        splits:       List of split names present in this version.
        total_images: Cached count of all images across splits.
        class_counts: Cached ``{class_name: count}`` mapping.
    """

    name = models.CharField(
        max_length=100,
        unique=True,
        validators=[MinLengthValidator(1)],
        help_text='Unique version identifier, e.g. "v1".',
    )
    parent = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='children',
        help_text='The parent version this was forked from.',
    )
    notes = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    # The active version is the one used for training / serving.
    # Only ONE version should be active at any time.
    is_active = models.BooleanField(
        default=False,
        db_index=True,
        help_text='Whether this is the active version for training.',
    )

    # Resolved split names (e.g. ["dataset_train", "dataset_test", "dataset_val"])
    splits = models.JSONField(default=list, blank=True)

    # Cached statistics (refreshed on version create / register)
    total_images = models.PositiveIntegerField(default=0)
    class_counts = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = 'dataset_versions'
        ordering = ['-created_at']
        verbose_name = 'Dataset version'
        verbose_name_plural = 'Dataset versions'

    def __str__(self) -> str:
        active = " ★" if self.is_active else ""
        return f"{self.name} ({self.total_images} images){active}"

    # -- helpers for delta-based versioning --

    def activate(self) -> None:
        """Mark this version as the active one for training.

        Deactivates all other versions first, ensuring only one
        version is active at any time.
        """
        DatasetVersion.objects.filter(is_active=True).update(is_active=False)
        self.is_active = True
        self.save(update_fields=['is_active'])

    @classmethod
    def get_active(cls) -> 'DatasetVersion | None':
        """Return the currently active dataset version, or None."""
        return cls.objects.filter(is_active=True).first()

    def refresh_cached_stats(self) -> None:
        """Recompute ``total_images``, ``class_counts``, ``splits`` from entries."""
        from django.db.models import Count
        entries = self.entries.all()

        self.total_images = entries.count()

        # {class_label: count}
        class_agg = (
            entries.values('class_label')
            .annotate(n=Count('id'))
            .order_by('class_label')
        )
        self.class_counts = {row['class_label']: row['n'] for row in class_agg}

        # unique splits
        split_names = (
            entries.values_list('split', flat=True)
            .distinct()
            .order_by('split')
        )
        self.splits = [s for s in split_names if s]

        self.save(update_fields=['total_images', 'class_counts', 'splits'])


class VersionEntry(models.Model):
    """One image's membership in a dataset version (delta-based).

    Instead of copying files on disk, each ``VersionEntry`` records:

    * **version** – which ``DatasetVersion`` owns this entry.
    * **physical_path** – absolute path to the image file on disk
      (e.g. ``datasets/v1/dataset_train/Plastic/img_001.jpg`` or
      ``media/uploads/2025/06/ood_shot.jpg``).
    * **split** – the logical split this image belongs to
      (``dataset_train``, ``dataset_test``, ``dataset_val``, or blank).
    * **class_label** – the class folder / operator-assigned label.
    * **filename** – human-readable basename (for display / pagination).
    * **source_image** – optional FK back to the OOD ``Image`` record
      that originated this entry (``NULL`` for seed-dataset files).

    When creating version *v(n+1)* from *v(n)*:

    1. Bulk-copy all ``VersionEntry`` rows of *v(n)*, pointing them at
       *v(n+1)* — this is a cheap DB ``INSERT … SELECT``, no file I/O.
    2. Create new ``VersionEntry`` rows for each staged OOD image.

    To "materialise" a version (e.g. for model training), a script can
    read all entries and symlink / copy on demand — but normal browsing
    and statistics are served purely from the DB.
    """

    version = models.ForeignKey(
        DatasetVersion,
        on_delete=models.CASCADE,
        related_name='entries',
        help_text='The dataset version this entry belongs to.',
    )

    # Where the actual file lives on disk (relative to project root).
    physical_path = models.CharField(
        max_length=500,
        help_text='Relative path to the image file from project root.',
    )

    split = models.CharField(
        max_length=50,
        blank=True,
        default='',
        db_index=True,
        help_text='Logical split (e.g. "dataset_train"). Blank for flat layouts.',
    )

    class_label = models.CharField(
        max_length=150,
        db_index=True,
        help_text='Class label (folder name or operator-assigned).',
    )

    filename = models.CharField(
        max_length=255,
        help_text='Image basename for display.',
    )

    file_size = models.PositiveIntegerField(
        default=0,
        help_text='Size of the physical file in bytes.',
    )

    source_image = models.ForeignKey(
        'Image',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='version_entries',
        help_text='The OOD Image record this entry originated from (NULL for seed data).',
    )

    added_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'version_entries'
        ordering = ['class_label', 'filename']
        verbose_name = 'Version entry'
        verbose_name_plural = 'Version entries'
        indexes = [
            # Fast lookup: all entries for a version + split + class
            models.Index(
                fields=['version', 'split', 'class_label'],
                name='idx_ve_version_split_class',
            ),
            # Fast lookup: all entries for a version (stats)
            models.Index(
                fields=['version', 'class_label'],
                name='idx_ve_version_class',
            ),
        ]
        # Prevent exact-duplicate entries within a version
        constraints = [
            models.UniqueConstraint(
                fields=['version', 'physical_path'],
                name='uq_ve_version_path',
            ),
        ]

    def __str__(self) -> str:
        return f"{self.version.name}/{self.split}/{self.class_label}/{self.filename}"


# ── Dataset class registry ──────────────────────────────────────────────────

class DatasetClass(models.Model):
    """Canonical registry of waste class labels.

    Grows automatically as operators label OOD images with new class names
    or as new dataset versions introduce new on-disk folders.

    Attributes:
        name:          Unique class label, e.g. ``"Plastic"``.
        introduced_in: The dataset version that first introduced this class
                       (``NULL`` for classes that came with the seed dataset).
    """

    name = models.CharField(
        max_length=150,
        unique=True,
        validators=[MinLengthValidator(1)],
        help_text='Canonical class label (e.g. "Plastic").',
    )
    created_at = models.DateTimeField(default=timezone.now)
    introduced_in = models.ForeignKey(
        DatasetVersion,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='introduced_classes',
        help_text='Version that first introduced this class.',
    )

    class Meta:
        db_table = 'dataset_classes'
        ordering = ['name']
        verbose_name = 'Dataset class'
        verbose_name_plural = 'Dataset classes'

    def __str__(self) -> str:
        return self.name


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

    # ── Dataset linkage ─────────────────────────────────────────────────
    added_to_dataset = models.BooleanField(default=False, db_index=True)
    dataset_version = models.ForeignKey(
        DatasetVersion,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='images',
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
            # Composite index for the staging area query
            models.Index(
                fields=['assigned_label', 'added_to_dataset'],
                name='idx_image_staging',
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

class TrashCounter(models.Model):
    """Running per-class item count, stored as a time-series.

    Each row is a snapshot: *"at <recorded_at>, class X had <total_count>
    items total"*.  The :meth:`increment` class-method atomically bumps
    the latest row or creates the first one.

    To query the *current* count for a class, fetch the most recent row::

        TrashCounter.objects.filter(class_name="Plastic").first()

    To query history, order by ``recorded_at``.
    """

    class_name = models.CharField(max_length=100, db_index=True)
    total_count = models.PositiveIntegerField()
    recorded_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = 'trash_counter'
        ordering = ['-recorded_at']
        verbose_name = 'Trash counter'
        verbose_name_plural = 'Trash counters'

    def __str__(self) -> str:
        return f"{self.class_name}: {self.total_count} ({self.recorded_at:%Y-%m-%d %H:%M})"

    @classmethod
    def increment(cls, class_name: str) -> None:
        """Atomically increment the count for *class_name*.

        Uses ``F()`` expressions to avoid race conditions.  A new row is
        appended to preserve the time-series history.
        """
        last = cls.objects.filter(class_name=class_name).first()
        new_count = (last.total_count + 1) if last else 1
        cls.objects.create(class_name=class_name, total_count=new_count)


# ── Training runs ───────────────────────────────────────────────────────────

class TrainingRun(models.Model):
    """Record of a model training / retraining run.

    Tracks the full lifecycle: config → data loading → training →
    evaluation → comparison → optional promotion.

    Status flow::

        pending → running → training → evaluating → completed
                                                   ↘ failed

    Each completed run produces artefacts under
    ``models/versions/<run_name>/`` (model checkpoint, metrics, logs).

    The ``is_active_model`` flag marks the run whose model is currently
    used for live inference.  Only one run should be active at a time.
    """

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('training', 'Training'),
        ('evaluating', 'Evaluating'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    # ── Identity ────────────────────────────────────────────────────────
    run_name = models.CharField(
        max_length=200, unique=True, db_index=True,
        help_text='Unique run identifier, e.g. "model_v2_20260224_143052".',
    )
    dataset_version_name = models.CharField(
        max_length=100, db_index=True,
        help_text='Name of the dataset version used for training.',
    )

    # ── Status ──────────────────────────────────────────────────────────
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default='pending', db_index=True,
    )
    error_message = models.TextField(blank=True, default='')

    # ── Configuration snapshot ──────────────────────────────────────────
    config = models.JSONField(
        default=dict, blank=True,
        help_text='Full training config snapshot (JSON).',
    )

    # ── Data stats ──────────────────────────────────────────────────────
    num_classes = models.PositiveIntegerField(null=True, blank=True)
    num_train_samples = models.PositiveIntegerField(null=True, blank=True)
    num_val_samples = models.PositiveIntegerField(null=True, blank=True)
    num_test_samples = models.PositiveIntegerField(null=True, blank=True)

    # ── Training progress ───────────────────────────────────────────────
    epochs_completed = models.PositiveIntegerField(default=0)

    # ── Evaluation results ──────────────────────────────────────────────
    test_accuracy = models.FloatField(null=True, blank=True)
    test_f1 = models.FloatField(null=True, blank=True)
    metrics_summary = models.JSONField(
        default=dict, blank=True,
        help_text='Top-level evaluation metrics (accuracy, F1, etc.).',
    )
    comparison = models.JSONField(
        default=dict, blank=True,
        help_text='Comparison with previous model (deltas + recommendation).',
    )

    # ── Model artefact ──────────────────────────────────────────────────
    model_path = models.CharField(
        max_length=500, blank=True, default='',
        help_text='Path to the saved .keras model file.',
    )
    is_active_model = models.BooleanField(
        default=False, db_index=True,
        help_text='Whether this model is currently serving live inference.',
    )

    # ── Timestamps ──────────────────────────────────────────────────────
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'training_runs'
        ordering = ['-created_at']
        verbose_name = 'Training run'
        verbose_name_plural = 'Training runs'

    def __str__(self) -> str:
        active = " ★" if self.is_active_model else ""
        return f"{self.run_name} ({self.status}){active}"

    @property
    def duration(self):
        """Return training duration as a timedelta, or None."""
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    def promote(self) -> None:
        """Make this run's model the active one for inference.

        Deactivates all other runs first.
        """
        TrainingRun.objects.filter(is_active_model=True).update(
            is_active_model=False,
        )
        self.is_active_model = True
        self.save(update_fields=['is_active_model'])
