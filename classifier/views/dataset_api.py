"""
Dataset versioning APIs — register, create, browse, and activate versions.

All versioning is **delta-based**: no files are duplicated on disk.
``VersionEntry`` rows track which images belong to which version.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from classifier.models import (
    DatasetClass,
    DatasetVersion,
    Image,
    VersionEntry,
)

from .helpers import (
    DATASETS_ROOT,
    IMG_EXTS,
    parse_json_body,
    safe_positive_int,
    scan_version_folder,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class registry
# ---------------------------------------------------------------------------

@require_GET
def api_classes(request):
    """Return the canonical list of all known waste class names."""
    classes = list(
        DatasetClass.objects.values(
            "id", "name", "introduced_in__name", "created_at",
        )
    )
    return JsonResponse({
        "classes": [
            {
                "id": c["id"],
                "name": c["name"],
                "introduced_in": c["introduced_in__name"],
                "created_at": (
                    c["created_at"].isoformat() if c["created_at"] else None
                ),
            }
            for c in classes
        ],
        "total": len(classes),
    })


# ---------------------------------------------------------------------------
# Version listing & activation
# ---------------------------------------------------------------------------

@require_GET
def api_dataset_versions(request):
    """List all registered dataset versions with cached statistics.

    Also returns the count of staged images (labelled but not yet added
    to any version).
    """
    versions = []
    for v in DatasetVersion.objects.all():
        versions.append({
            "id": v.id,
            "name": v.name,
            "parent": v.parent.name if v.parent else None,
            "notes": v.notes,
            "created_at": v.created_at.isoformat(),
            "total_images": v.total_images,
            "class_counts": v.class_counts or {},
            "classes": sorted((v.class_counts or {}).keys()),
            "splits": v.splits or [],
            "is_active": v.is_active,
        })

    staged = Image.objects.filter(
        assigned_label__isnull=False,
        added_to_dataset=False,
    ).count()

    return JsonResponse({"versions": versions, "staged_count": staged})


@require_GET
def api_active_version(request):
    """Return the currently active dataset version for training.

    Returns the active version's details, or ``null`` if none is set.
    """
    active = DatasetVersion.get_active()
    if not active:
        return JsonResponse({"active_version": None})

    return JsonResponse({
        "active_version": {
            "id": active.id,
            "name": active.name,
            "parent": active.parent.name if active.parent else None,
            "total_images": active.total_images,
            "class_counts": active.class_counts or {},
            "classes": sorted((active.class_counts or {}).keys()),
            "splits": active.splits or [],
            "created_at": active.created_at.isoformat(),
        },
    })


@csrf_exempt
@require_POST
def api_set_active_version(request):
    """Set a specific dataset version as the active one for training.

    Request body (JSON):
        name : str — the version name to activate (required).
    """
    body = parse_json_body(request)
    name = body.get("name", "").strip()

    if not name:
        return JsonResponse(
            {"error": "Version name is required."}, status=400,
        )

    try:
        version_obj = DatasetVersion.objects.get(name=name)
    except DatasetVersion.DoesNotExist:
        return JsonResponse(
            {"error": f"Version '{name}' not found."}, status=404,
        )

    version_obj.activate()

    logger.info("Activated dataset version '%s'", name)
    return JsonResponse({
        "id": version_obj.id,
        "name": version_obj.name,
        "is_active": True,
        "total_images": version_obj.total_images,
    })


# ---------------------------------------------------------------------------
# Staging area
# ---------------------------------------------------------------------------

@require_GET
def api_staged_images(request):
    """List OOD images that are labelled but not yet added to a version.

    These images sit in a staging area waiting to be included in the
    next dataset version via ``api_create_version``.
    """
    images = Image.objects.filter(
        assigned_label__isnull=False,
        added_to_dataset=False,
    ).order_by("-uploaded_at")

    results = [
        {
            "id": img.id,
            "filename": img.filename,
            "image_url": img.image.url if img.image else None,
            "assigned_label": img.assigned_label,
            "uploaded_at": img.uploaded_at.isoformat(),
        }
        for img in images
    ]

    label_counts: Dict[str, int] = {}
    for r in results:
        lbl = r["assigned_label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    return JsonResponse({
        "images": results,
        "total": len(results),
        "label_counts": label_counts,
    })


# ---------------------------------------------------------------------------
# Create new version (delta-based)
# ---------------------------------------------------------------------------

@csrf_exempt
@require_POST
def api_create_version(request):
    """Create a new dataset version using delta-based tracking.

    **No files are copied on disk.**  Instead:

    1. If a parent version is specified, all of its ``VersionEntry``
       rows are bulk-duplicated with the new version as owner.
    2. Each staged OOD image gets a new ``VersionEntry`` pointing at
       its *existing* physical location (``media/uploads/…``).
    3. Cached statistics on the ``DatasetVersion`` are refreshed.

    Request body (JSON):
        name   : str — unique version name (required).
        parent : str — name of the parent version to inherit from (optional).
        notes  : str — free-text description.
    """
    body = parse_json_body(request)
    name = body.get("name", "").strip()
    parent_name = body.get("parent", "").strip()
    notes = body.get("notes", "")

    if not name:
        return JsonResponse(
            {"error": "Version name is required."}, status=400,
        )

    if DatasetVersion.objects.filter(name=name).exists():
        return JsonResponse(
            {"error": f"Version '{name}' already exists."}, status=400,
        )

    # ── Resolve parent ──────────────────────────────────────────────
    parent_version = None
    if parent_name:
        try:
            parent_version = DatasetVersion.objects.get(name=parent_name)
        except DatasetVersion.DoesNotExist:
            return JsonResponse(
                {"error": f"Parent version '{parent_name}' not found."},
                status=404,
            )

    staged = Image.objects.filter(
        assigned_label__isnull=False,
        added_to_dataset=False,
    )
    staged_count = staged.count()

    if staged_count == 0 and not parent_version:
        return JsonResponse(
            {"error": "No staged images and no parent to inherit from."},
            status=400,
        )

    # ── Create the version record ───────────────────────────────────
    version_obj = DatasetVersion.objects.create(
        name=name, parent=parent_version, notes=notes,
    )

    # ── Step 1: Inherit parent entries (cheap DB-only bulk insert) ──
    inherited_count = 0
    if parent_version:
        parent_entries = parent_version.entries.all()
        new_entries = []
        for entry in parent_entries.iterator(chunk_size=2000):
            new_entries.append(VersionEntry(
                version=version_obj,
                physical_path=entry.physical_path,
                split=entry.split,
                class_label=entry.class_label,
                filename=entry.filename,
                file_size=entry.file_size,
                source_image=entry.source_image,
            ))
            if len(new_entries) >= 2000:
                VersionEntry.objects.bulk_create(
                    new_entries, ignore_conflicts=True,
                )
                inherited_count += len(new_entries)
                new_entries = []
        if new_entries:
            VersionEntry.objects.bulk_create(
                new_entries, ignore_conflicts=True,
            )
            inherited_count += len(new_entries)

    # ── Step 2: Add staged OOD images as new entries ────────────────
    parent_splits = parent_version.splits if parent_version else []
    target_split = ""
    if parent_splits:
        target_split = next(
            (s for s in parent_splits if "train" in s.lower()),
            parent_splits[0],
        )

    new_staged_entries = []
    for img in staged:
        label = img.assigned_label
        physical = str(img.image)       # relative to MEDIA_ROOT
        media_rel = f"media/{physical}"

        DatasetClass.objects.get_or_create(
            name=label, defaults={"introduced_in": version_obj},
        )

        new_staged_entries.append(VersionEntry(
            version=version_obj,
            physical_path=media_rel,
            split=target_split,
            class_label=label,
            filename=img.filename or Path(physical).name,
            file_size=img.file_size or 0,
            source_image=img,
        ))

        img.added_to_dataset = True
        img.dataset_version = version_obj
        img.save(update_fields=["added_to_dataset", "dataset_version"])

    if new_staged_entries:
        VersionEntry.objects.bulk_create(
            new_staged_entries, ignore_conflicts=True,
        )

    # ── Step 3: Refresh cached stats ────────────────────────────────
    version_obj.refresh_cached_stats()

    # ── Step 4: Auto-activate the newest version ────────────────────
    version_obj.activate()

    logger.info(
        "Created dataset version '%s' (inherited %d, staged %d, total %d)",
        name, inherited_count, staged_count, version_obj.total_images,
    )

    return JsonResponse({
        "id": version_obj.id,
        "name": version_obj.name,
        "parent": parent_name or None,
        "total_images": version_obj.total_images,
        "class_counts": version_obj.class_counts,
        "splits": version_obj.splits,
        "staged_added": staged_count,
        "inherited": inherited_count,
        "is_active": version_obj.is_active,
    })


# ---------------------------------------------------------------------------
# Register existing on-disk folder
# ---------------------------------------------------------------------------

@csrf_exempt
@require_POST
def api_register_version(request):
    """Register an existing on-disk folder as a dataset version.

    Scans the folder structure, creates ``DatasetVersion`` +
    ``DatasetClass`` records, and populates ``VersionEntry`` rows for
    every image found.

    Request body (JSON):
        name   : str — folder name under ``datasets/`` (required).
        notes  : str — free-text description.
        parent : str — name of the parent version (optional).
    """
    body = parse_json_body(request)
    name = body.get("name", "").strip()
    notes = body.get("notes", "")
    parent_name = body.get("parent", "").strip()

    if not name:
        return JsonResponse(
            {"error": "Version name is required."}, status=400,
        )

    version_path = DATASETS_ROOT / name
    if not version_path.exists():
        return JsonResponse(
            {"error": f"Folder '{name}' not found in datasets/."}, status=404,
        )

    if DatasetVersion.objects.filter(name=name).exists():
        return JsonResponse(
            {"error": f"Version '{name}' is already registered."}, status=400,
        )

    parent_version = None
    if parent_name:
        parent_version = DatasetVersion.objects.filter(
            name=parent_name,
        ).first()

    # Scan the on-disk folder structure
    splits, counts, total = scan_version_folder(version_path)

    version_obj = DatasetVersion.objects.create(
        name=name,
        parent=parent_version,
        notes=notes,
        total_images=total,
        class_counts=counts,
        splits=splits,
    )

    # ── Create VersionEntry rows for every image on disk ────────────
    entries_to_create: List[VersionEntry] = []

    if splits:
        for split_name in splits:
            split_dir = version_path / split_name
            if not split_dir.is_dir():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for f in sorted(cls_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in IMG_EXTS:
                        rel_path = f.relative_to(Path(settings.BASE_DIR))
                        entries_to_create.append(VersionEntry(
                            version=version_obj,
                            physical_path=str(rel_path).replace("\\", "/"),
                            split=split_name,
                            class_label=cls_dir.name,
                            filename=f.name,
                            file_size=f.stat().st_size,
                        ))
                        if len(entries_to_create) >= 2000:
                            VersionEntry.objects.bulk_create(
                                entries_to_create, ignore_conflicts=True,
                            )
                            entries_to_create = []
    else:
        for cls_dir in sorted(version_path.iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    rel_path = f.relative_to(Path(settings.BASE_DIR))
                    entries_to_create.append(VersionEntry(
                        version=version_obj,
                        physical_path=str(rel_path).replace("\\", "/"),
                        split="",
                        class_label=cls_dir.name,
                        filename=f.name,
                        file_size=f.stat().st_size,
                    ))
                    if len(entries_to_create) >= 2000:
                        VersionEntry.objects.bulk_create(
                            entries_to_create, ignore_conflicts=True,
                        )
                        entries_to_create = []

    if entries_to_create:
        VersionEntry.objects.bulk_create(entries_to_create, ignore_conflicts=True)

    # ── Create DatasetClass records ─────────────────────────────────
    new_classes: List[str] = []
    for cls_name in sorted(counts.keys()):
        _, created = DatasetClass.objects.get_or_create(
            name=cls_name, defaults={"introduced_in": version_obj},
        )
        if created:
            new_classes.append(cls_name)

    # Auto-activate if this is the first version registered
    if DatasetVersion.objects.count() == 1:
        version_obj.activate()

    logger.info(
        "Registered dataset version '%s' (%d images, %d entries, %d new classes)",
        name, total, version_obj.entries.count(), len(new_classes),
    )

    return JsonResponse({
        "id": version_obj.id,
        "name": version_obj.name,
        "total_images": total,
        "class_counts": counts,
        "splits": splits,
        "classes_created": new_classes,
        "is_active": version_obj.is_active,
    })


# ---------------------------------------------------------------------------
# Browse images within a version
# ---------------------------------------------------------------------------

@require_GET
def api_dataset_images(request):
    """Browse images inside a specific version / split / class.

    Query params
    ------------
    version  : str — version name (required).
    split    : str — split name, e.g. ``"dataset_train"`` (optional).
    class    : str — class label (required).
    page     : int — page number (default 1).
    per_page : int — items per page (default 24).
    """
    version = request.GET.get("version", "").strip()
    split = request.GET.get("split", "").strip()
    label = request.GET.get("class", "").strip()
    page = safe_positive_int(request.GET.get("page"), 1)
    per_page = safe_positive_int(request.GET.get("per_page"), 24)

    if not version or not label:
        return JsonResponse(
            {"error": "Both 'version' and 'class' query params are required."},
            status=400,
        )

    try:
        version_obj = DatasetVersion.objects.get(name=version)
    except DatasetVersion.DoesNotExist:
        return JsonResponse(
            {"error": f"Version '{version}' not found."}, status=404,
        )

    entries = VersionEntry.objects.filter(
        version=version_obj,
        class_label=label,
    )
    if split:
        entries = entries.filter(split=split)

    entries = entries.order_by("filename")
    total = entries.count()
    pages = (total + per_page - 1) // per_page if total > 0 else 0
    start = (page - 1) * per_page
    page_entries = entries[start : start + per_page]

    images: List[Dict[str, Any]] = []
    for entry in page_entries:
        images.append({
            "filename": entry.filename,
            "path": entry.physical_path,
            "size": entry.file_size,
        })

    return JsonResponse({
        "images": images,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
        "version": version,
        "split": split,
        "class": label,
    })
