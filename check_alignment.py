"""Verify DB <-> Disk alignment and split tracking."""
import os, pathlib, warnings
os.environ["DJANGO_SETTINGS_MODULE"] = "wastex.settings"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import django; django.setup()
from classifier.models import DatasetVersion, VersionEntry, DatasetClass

print("=" * 60)
print("1. VersionEntry TABLE SCHEMA - do we track splits?")
print("=" * 60)
for f in VersionEntry._meta.get_fields():
    print(f"  {f.name:<20} {f.__class__.__name__}")

print()
print("=" * 60)
print("2. SAMPLE VersionEntry ROWS (first 5)")
print("=" * 60)
for e in VersionEntry.objects.filter(version__name="v1")[:5]:
    print(f"  split={e.split:<20} class={e.class_label:<20} file={e.physical_path}")

print()
print("=" * 60)
print("3. SPLIT-LEVEL ALIGNMENT - DB vs Disk")
print("=" * 60)

root = pathlib.Path("datasets/v1")
splits = ["dataset_train", "dataset_val", "dataset_test"]

print(f"  {'Split':<20} {'DB':>8} {'Disk':>8} {'Match':>8}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

total_db = total_disk = 0
all_ok = True
for s in splits:
    db_count = VersionEntry.objects.filter(version__name="v1", split=s).count()
    disk_count = sum(1 for f in (root / s).rglob("*") if f.is_file())
    total_db += db_count
    total_disk += disk_count
    match = "YES" if db_count == disk_count else "NO"
    if db_count != disk_count:
        all_ok = False
    print(f"  {s:<20} {db_count:>8} {disk_count:>8} {match:>8}")

print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'TOTAL':<20} {total_db:>8} {total_disk:>8} {'YES' if total_db == total_disk else 'NO':>8}")

print()
print("=" * 60)
print("4. CLASS-LEVEL ALIGNMENT - DB vs Disk (per split)")
print("=" * 60)

classes = sorted(VersionEntry.objects.filter(version__name="v1")
                 .values_list("class_label", flat=True).distinct())
mismatches = []
for s in splits:
    print(f"\n  {s}:")
    print(f"    {'Class':<20} {'DB':>6} {'Disk':>6} {'OK':>6}")
    print(f"    {'-'*20} {'-'*6} {'-'*6} {'-'*6}")
    for cls in classes:
        db_n = VersionEntry.objects.filter(version__name="v1", split=s, class_label=cls).count()
        cls_dir = root / s / cls
        disk_n = sum(1 for f in cls_dir.rglob("*") if f.is_file()) if cls_dir.exists() else 0
        ok = "YES" if db_n == disk_n else "NO"
        if db_n != disk_n:
            mismatches.append((s, cls, db_n, disk_n))
        print(f"    {cls:<20} {db_n:>6} {disk_n:>6} {ok:>6}")

print()
print("=" * 60)
print("5. SPOT CHECK - can data.py resolve these file paths?")
print("=" * 60)

base = pathlib.Path(".").resolve()
sample = VersionEntry.objects.filter(version__name="v1")[:10]
for e in sample:
    full = base / e.physical_path
    exists = full.exists()
    status = "EXISTS" if exists else "MISSING"
    print(f"  [{status}] {e.physical_path}")

print()
print("=" * 60)
print("6. DatasetClass TABLE vs VersionEntry labels")
print("=" * 60)
db_classes = set(DatasetClass.objects.values_list("name", flat=True))
entry_classes = set(VersionEntry.objects.filter(version__name="v1")
                    .values_list("class_label", flat=True).distinct())
print(f"  DatasetClass table : {sorted(db_classes)}")
print(f"  VersionEntry labels: {sorted(entry_classes)}")
in_db_not_entries = sorted(db_classes - entry_classes)
in_entries_not_db = sorted(entry_classes - db_classes)
if in_db_not_entries:
    print(f"  EXTRA in DatasetClass (not in v1 entries): {in_db_not_entries}")
if in_entries_not_db:
    print(f"  MISSING from DatasetClass: {in_entries_not_db}")
if not in_db_not_entries and not in_entries_not_db:
    print(f"  Perfect match!")

print()
print("=" * 60)
if mismatches:
    print("RESULT: MISMATCHES FOUND:")
    for s, c, db_n, disk_n in mismatches:
        print(f"  {s}/{c}: DB={db_n}, Disk={disk_n}")
elif not all_ok:
    print("RESULT: SPLIT TOTALS DON'T MATCH")
else:
    print("RESULT: EVERYTHING ALIGNED - DB and Disk match perfectly")
print("=" * 60)
