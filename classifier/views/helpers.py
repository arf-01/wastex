"""
Shared constants, utilities, and helper functions used across views.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from django.conf import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES_FILE = Path(settings.BASE_DIR) / "models" / "classes.txt"
MODEL_CLASS_NAMES: List[str] = [
    line.strip() for line in open(CLASSES_FILE) if line.strip()
]

DATASETS_ROOT = Path(settings.DATASETS_ROOT)

IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_CONTENT_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
})


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_all_class_names() -> List[str]:
    """Return the canonical class-name list from the database."""
    from classifier.models import DatasetClass

    return list(DatasetClass.objects.values_list("name", flat=True))


def parse_json_body(request) -> Dict[str, Any]:
    """Safely parse a JSON request body, returning {} on failure."""
    try:
        return json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def paginate(items: list, page: int, per_page: int) -> Tuple[list, int, int]:
    """Slice *items* into a page and return ``(page_items, total, pages)``."""
    total = len(items)
    pages = (total + per_page - 1) // per_page if total > 0 else 0
    start = (page - 1) * per_page
    return items[start : start + per_page], total, pages


def safe_positive_int(value: Any, default: int) -> int:
    """Coerce *value* to a positive int, falling back to *default*."""
    try:
        n = int(value)
        return n if n > 0 else default
    except (TypeError, ValueError):
        return default


def scan_version_folder(
    version_path: Path,
) -> Tuple[List[str], Dict[str, int], int]:
    """Scan a dataset version folder and return split-aware class counts.

    Used **only** during initial registration of an existing on-disk
    dataset (e.g. ``datasets/v1``).  After registration, all lookups
    are served from ``VersionEntry`` rows in the database.

    Supports two on-disk layouts:

    Split layout (preferred)::

        version/dataset_train/Plastic/  ← images
        version/dataset_test/Glass/     ← images

    Flat layout (legacy)::

        version/Plastic/  ← images
        version/Glass/    ← images

    Returns
    -------
    (splits, class_counts, total)
        *splits* is a sorted list of split folder names,
        *class_counts* maps class name → image count (summed across splits),
        *total* is the overall image count.
    """
    splits: List[str] = []
    class_counts: Dict[str, int] = {}
    total = 0

    if not version_path.exists():
        return splits, class_counts, total

    first_level_dirs = sorted(d for d in version_path.iterdir() if d.is_dir())
    if not first_level_dirs:
        return splits, class_counts, total

    # Detect split vs flat: split dirs contain class sub-dirs.
    is_split_layout = any(
        any(sd.is_dir() for sd in d.iterdir()) for d in first_level_dirs
    )

    if is_split_layout:
        for split_dir in first_level_dirs:
            if not split_dir.is_dir():
                continue
            splits.append(split_dir.name)
            for cls_dir in sorted(split_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                n = sum(
                    1
                    for f in cls_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMG_EXTS
                )
                class_counts[cls_dir.name] = (
                    class_counts.get(cls_dir.name, 0) + n
                )
                total += n
    else:
        for cls_dir in first_level_dirs:
            if not cls_dir.is_dir():
                continue
            n = sum(
                1
                for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTS
            )
            class_counts[cls_dir.name] = n
            total += n

    return sorted(splits), class_counts, total
