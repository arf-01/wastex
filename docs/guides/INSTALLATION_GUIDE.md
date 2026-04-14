# WasteX Installation Guide

**For Thesis Submission & Deployment**

---

## Quick Start (Windows)

### Option A: Automated Installation (Recommended)

```bash
# 1. Double-click install.bat
install.bat

# 2. Follow the prompts:
#    - Select data folder (e.g., D:\WasteX)
#    - Confirm installation
#    - Select "Yes" to start server

# 3. Browser opens to http://localhost:8000/
```

### Option B: Manual Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize database
python manage.py migrate

# 3. Configure storage paths (choose ONE method):

# Method 1: Using environment variables
set WASTE_MEDIA_ROOT=D:\WasteX\media
set WASTE_DATASETS_ROOT=D:\WasteX\datasets
set WASTE_MODELS_ROOT=D:\WasteX\models

# Method 2: Using initialization command
python manage.py initialize_paths ^
    --media-root "D:\WasteX\media" ^
    --datasets-root "D:\WasteX\datasets" ^
    --models-root "D:\WasteX\models"

# 4. Start the server
python manage.py runserver

# 5. Open browser: http://localhost:8000/
```

---

## Storage Configuration

### Where are files stored?

After installation, WasteX stores files in three locations:

```
User-Selected Data Folder (e.g., D:\WasteX\)
├── media/                          ← Uploaded images
│   └── uploads/YYYY/MM/DD/
│       ├── image_001.jpg
│       ├── image_002.jpg
│       └── ...
├── datasets/                       ← Training data (organized by version)
│   ├── v1/
│   │   ├── train/
│   │   │   ├── Plastic/
│   │   │   ├── Glass/
│   │   │   └── ...
│   │   ├── val/
│   │   ├── test/
│   │   └── metadata.json
│   └── v2/
│       └── ...
└── models/                         ← Trained ML models
    ├── logits_mdl.keras            ← Current active model
    ├── classes.txt
    └── versions/
        ├── v1_20260224_162046/
        │   ├── model.keras
        │   ├── config.json
        │   └── metrics.json
        └── ...
```

### System Requirements

- **OS:** Windows 7 or later, macOS 10.13+, or Linux
- **Python:** 3.9 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 
  - Installation: 2GB (for Python + dependencies)
  - Data: Varies (50GB+ recommended for datasets)
- **Internet:** Required for first installation (pip packages), optional thereafter

### Choosing a Data Folder

**Recommended:** Separate drive with sufficient space

```
❌ Poor choice:
   C:\WasteX\
   → C: drive often limited, system files conflict

✅ Good choices:
   D:\WasteX\
   → Dedicated second drive

✅ Better choices:
   D:\Projects\WasteClassification\
   → Clearly named, easy to back up
   → Can expand easily to other drives

✅ Network paths (use with caution):
   \\server\shared\WasteX\
   → Slower than local disk
   → Requires network connectivity
   → Not recommended for training
```

---

## Installation Methods

### Method 1: Windows Installer (For Non-Technical Users)

```
1. Download: WasteX-v2.0-Setup.exe
2. Double-click
3. Select installation location
4. Select data folder
5. Finish → App launches automatically
```

Status: **Planned** (can be built with NSIS or Inno Setup)

### Method 2: Batch File (Current, For Tech Users)

```bash
# Windows:
install.bat

# macOS/Linux:
./install.sh
```

### Method 3: Command Line (Manual Control)

```bash
# Full installation with full control
pip install -r requirements.txt
python manage.py migrate
python manage.py initialize_paths --media-root "YOUR_PATH/media" ...
python manage.py runserver
```

---

## Configuration

### During Installation

Storage paths are set **once** during installation:

1. **Detection:** WasteX checks available drives
2. **Selection:** User chooses folder
3. **Validation:**
   - Path must be writable
   - Must have 50GB+ free space
   - Network paths get a warning
4. **Storage:** Paths saved to database (`AppSettings` table)

### After Installation

To change paths, **reinstall** with new settings:

```bash
# Backup old data first!
# Then reinstall:
python manage.py initialize_paths --media-root "NEW_PATH/media" ...
```

---

## Troubleshooting

### "No write permission for D:\WasteX\"

```
Problem: User selected a read-only folder
Solution:
  1. Choose different folder (user profile, external drive)
  2. Change folder permissions (Windows Settings → Security)
  3. Run as Administrator
```

### "Insufficient disk space. Required: 50GB, Available: 5GB"

```
Problem: Drive too small for training data
Solution:
  1. Use larger drive (internal or external)
  2. Reduce dataset size
  3. Train with smaller batches
```

### "Cannot access path: C:\Users\Username\AppData"

```
Problem: Path with special characters or permissions
Solution:
  1. Reinstall with simpler path (e.g., D:\WasteX)
  2. Run installer as Administrator
  3. Use network path (if available)
```

### "Database locked" or "File cannot be written"

```
Problem: Two instances of WasteX running
Solution:
  1. Check for other WasteX windows/processes
  2. Use Task Manager to kill Python processes
  3. Restart WasteX
```

---

## Backup & Recovery

### Backup Data

```bash
# Backup configured data folder:
# E.g., if data is at D:\WasteX\

xcopy D:\WasteX\ E:\Backup\WasteX\ /E /I /Y

# Backup database (SQLite):
copy db.sqlite3 backup\db.sqlite3.backup
```

### Restore Data

```bash
# Reinstall WasteX
python manage.py migrate

# Copy backup to configured folder:
xcopy E:\Backup\WasteX\ D:\WasteX\ /E /I /Y

# Restore database:
copy backup\db.sqlite3.backup db.sqlite3
```

---

## For Development/Testing

### Override Configuration with Environment Variables

```bash
# Temporary override (current session only):
set WASTE_MEDIA_ROOT=C:\temp\media
set WASTE_DATASETS_ROOT=C:\temp\datasets
set WASTE_MODELS_ROOT=C:\temp\models

python manage.py runserver
```

### Multiple Installations (Different Folders)

```bash
# Installation 1: Production
python manage.py initialize_paths --media-root "D:\WasteX_Prod\media" ...

# Installation 2: Testing
python manage.py initialize_paths --media-root "E:\WasteX_Test\media" ...
```

---

## Architecture & Design

### Why Fixed Paths?

Fixed paths set during installation:
- ✅ **Simplicity:** No runtime path changes
- ✅ **Reliability:** No crashes due to missing paths
- ✅ **Predictability:** Always know where files are
- ✅ **Professional:** Enterprise software standard

### How Paths Are Used

1. **Django Settings** (`wastex/settings.py`):
   ```python
   MEDIA_ROOT = get_storage_path('WASTE_MEDIA_ROOT', 'media_root', 'media')
   ```

2. **AppSettings Database Model** (`classifier/models.py`):
   ```python
   AppSettings.get('media_root')  # Returns: 'D:/WasteX/media'
   ```

3. **All File Operations Use These Paths:**
   ```python
   image_path = MEDIA_ROOT / 'uploads' / '2026' / '03' / '29' / 'image.jpg'
   model_path = MODELS_ROOT / 'logits_mdl.keras'
   dataset_path = DATASETS_ROOT / 'v1' / 'train' / 'Plastic' / 'image.jpg'
   ```

### Database Schema

```
AppSettings table:
┌───────────────────────────────────────────────────┐
│ key            │ value                │ description │
├────────────────┼──────────────────────┼─────────────┤
│ media_root     │ D:/WasteX/media      │ Uploaded... │
│ datasets_root  │ D:/WasteX/datasets   │ Training... │
│ models_root    │ D:/WasteX/models     │ ML models   │
└───────────────────────────────────────────────────┘
```

---

## Performance Considerations

### Local Drive vs Network Drive

| Metric | Local (D:) | Network (\\server\) |
|--------|-----------|-------------------|
| **Upload Speed** | ~100MB/s | ~10MB/s |
| **Training Speed** | 1 image/0.3s | 1 image/3s |
| **Reliability** | High | Medium (network issues) |
| **Disk Space** | Limited | Unlimited (shared) |
| **Cost** | External drive ~$100 | Existing infrastructure |

**Recommendation:** Use local drive for primary data, network for backups.

---

## For Thesis Documentation

### What to Include in Thesis

**Section: Installation & Configuration**

```markdown
### 5.2 Storage Configuration

WasteX uses configurable storage paths, allowing operators
to choose where data is stored based on their hardware setup.

**Installation Process:**
1. User runs installer or initialization command
2. System prompts for storage folder location
3. Paths are validated (write permission, disk space)
4. Configuration is saved to database

**Design Rationale:**
Fixed paths (set at installation, not changeable at runtime)
provide:
- Simplicity (no runtime path resolution)
- Reliability (failures are caught early)
- Predictability (always know where files are)

This design is standard in enterprise software and aligns
with Django best practices.

**Supported Paths:**
- Local drives: C:, D:, E: (any fixed drive)
- External drives: USB, external HDDs
- Network paths: UNC paths (\\server\share) with caution

**Folder Structure:**
All three path types can be on same or different drives,
allowing operators to optimize storage utilization.
```

### What to Show in Thesis Presentation

- Installation output (screenshot)
- Before/After disk space (show configuration works)
- Backup of configured paths (show it's practical)

---

## Support & Contact

For issues:
1. Check [Troubleshooting](#troubleshooting) above
2. Run: `python manage.py check`
3. Check database: `python manage.py shell` → `AppSettings.objects.all()`

---

**Installation Guide Version:** 2.0  
**Last Updated:** April 10, 2026  
**For:** WasteX Thesis Project
