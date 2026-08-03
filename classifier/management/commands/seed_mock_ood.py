"""
Management command: seed_mock_ood
---------------------------------
Inserts mock OOD Image records so the Edge OOD dashboard has data to display.
No real image files are required — a small placeholder JPEG is generated in
memory and saved through Django's storage backend.

Usage:
    python manage.py seed_mock_ood            # add 10 records (default)
    python manage.py seed_mock_ood --count 25
    python manage.py seed_mock_ood --clear    # wipe existing mock data first
"""

from __future__ import annotations

import io
import random
from datetime import timedelta

from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from django.utils import timezone

from classifier.models import Image


# ── Palette of realistic-looking mock values ────────────────────────────────

FILENAMES = [
    "capture_20260704_143201_001.jpg",
    "capture_20260704_151832_002.jpg",
    "capture_20260704_160045_003.jpg",
    "capture_20260705_090112_004.jpg",
    "capture_20260705_093347_005.jpg",
    "capture_20260705_102201_006.jpg",
    "capture_20260705_110539_007.jpg",
    "capture_20260705_115823_008.jpg",
    "capture_20260705_124410_009.jpg",
    "capture_20260705_133001_010.jpg",
    "capture_20260705_141730_011.jpg",
    "capture_20260705_152209_012.jpg",
    "capture_20260705_161845_013.jpg",
    "capture_20260706_083312_014.jpg",
    "capture_20260706_092507_015.jpg",
    "capture_20260706_100933_016.jpg",
    "capture_20260706_113421_017.jpg",
    "capture_20260706_121054_018.jpg",
    "capture_20260706_134716_019.jpg",
    "capture_20260706_143228_020.jpg",
    "capture_20260706_152901_021.jpg",
    "capture_20260706_161533_022.jpg",
    "capture_20260707_090044_023.jpg",
    "capture_20260707_101212_024.jpg",
    "capture_20260707_113347_025.jpg",
]

# OOD images have no top_prediction — but we store the energy score
# and logits in all_predictions as the real inference pipeline does.
def _make_predictions(num_classes: int = 6) -> dict:
    logits = [round(random.uniform(-3.5, 1.2), 4) for _ in range(num_classes)]
    energy = round(-1.0 * __import__('math').log(
        sum(__import__('math').exp(l) for l in logits)
    ), 6)
    return {"logits": logits, "energy": energy}


def _tiny_jpeg() -> bytes:
    """Return a minimal valid JPEG (1×1 grey pixel) as bytes."""
    # Minimal JPEG binary for a 1x1 grey pixel
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00,
        0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB,
        0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07,
        0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B,
        0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E,
        0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C,
        0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34,
        0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34,
        0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01,
        0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01,
        0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
        0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
        0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
        0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1,
        0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
        0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36,
        0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64,
        0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77,
        0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A,
        0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4,
        0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8,
        0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
        0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1,
        0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA,
        0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD4, 0xFF,
        0xD9,
    ])


class Command(BaseCommand):
    help = "Seed mock OOD Image records for dashboard testing."

    def add_arguments(self, parser):
        parser.add_argument(
            "--count",
            type=int,
            default=10,
            help="Number of mock OOD records to create (default: 10).",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete all existing OOD Image records before seeding.",
        )

    def handle(self, *args, **options):
        count: int = options["count"]
        clear: bool = options["clear"]

        if clear:
            deleted, _ = Image.objects.filter(top_prediction__isnull=True).delete()
            self.stdout.write(self.style.WARNING(
                f"Cleared {deleted} existing OOD record(s)."
            ))

        # We will fetch a new image for each record, or fallback to tiny jpeg if it fails.
        import urllib.request
        def _download_mock_image() -> bytes:
            try:
                url = "https://picsum.photos/400/300"
                with urllib.request.urlopen(url, timeout=5) as response:
                    return response.read()
            except Exception as e:
                return _tiny_jpeg()

        now = timezone.now()
        created = 0

        filenames = (FILENAMES * ((count // len(FILENAMES)) + 1))[:count]
        random.shuffle(filenames)

        for i, fname in enumerate(filenames):
            # Spread uploads over the past 7 days for a realistic timeline
            offset_minutes = random.randint(0, 60 * 24 * 7)
            upload_time = now - timedelta(minutes=offset_minutes)

            predictions = _make_predictions()
            confidence = round(random.uniform(0.3, 0.69), 4)  # below 0.7 threshold

            img = Image(
                filename=fname,
                top_prediction=None,          # NULL → flagged as OOD
                confidence=confidence,
                all_predictions=predictions,
                reviewed=False,               # Force to unreviewed so it shows in pending
                uploaded_at=upload_time,

                classified_at=upload_time,
            )

            # Save a downloaded placeholder image through Django's storage
            img_bytes = _download_mock_image()
            img.image.save(
                fname,
                ContentFile(img_bytes),
                save=False,
            )
            img.save()
            created += 1

            self.stdout.write(
                f"  [{i+1:>3}/{count}] Created OOD record: {fname} "
                f"(energy={predictions['energy']:.4f}, conf={confidence:.4f})"
            )

        self.stdout.write(self.style.SUCCESS(
            f"\nDone -- created {created} mock OOD record(s). "
            f"Visit /edge/dashboard/ or your OOD inspect page to see them."
        ))
