# Installation System: Visual Summary

Complete visual guide for understanding and explaining the installation system.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   WasteX Installation System                │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   User Runs      │
│  install.bat     │
└────────┬─────────┘
         │
         ├─ Check Python installed
         ├─ Ask: Where to store data?
         │
         └──────────────────┐
                            │
                            ▼
              ┌──────────────────────────┐
              │  User Selects Folder     │
              │  E.g., D:\WasteX         │
              └────────┬─────────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │  System Validates Path:    │
          ├────────────────────────────┤
          │ ✓ Create folder if missing │
          │ ✓ Check write permission   │
          │ ✓ Check 50GB+ free space   │
          │ ✓ Test write/delete file   │
          └────────┬───────────────────┘
                   │
         ┌─────────┴─────────┐
         │ Validation OK?    │
         ├─────────┬─────────┤
        YES       NO        │
         │         │        │
         │        Show Error │
         │         Retry     │
         │                   │
         └────────┬──────────┘
                  │
                  ▼
        ┌─────────────────────────────┐
        │  Save to Database           │
        │                             │
        │ AppSettings.set(            │
        │   'media_root',             │
        │   'D:\WasteX\media'         │
        │ )                           │
        │                             │
        │ AppSettings.set(            │
        │   'datasets_root',          │
        │   'D:\WasteX\datasets'      │
        │ )                           │
        │                             │
        │ AppSettings.set(            │
        │   'models_root',            │
        │   'D:\WasteX\models'        │
        │ )                           │
        └────────┬────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  Start Server      │
        │                    │
        │  python manage.py  │
        │  runserver         │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────────────┐
        │  Django Loads Settings     │
        │                            │
        │  MEDIA_ROOT =              │
        │    get_storage_path(...)   │
        │      → Query AppSettings   │
        │      → Get 'D:\WasteX\...' │
        └────────┬───────────────────┘
                 │
                 ▼
        ┌────────────────────────────┐
        │  All File Operations       │
        │  Use Configured Paths      │
        │                            │
        │  Image upload → MEDIA_ROOT │
        │  Training data → DATASETS  │
        │  Models → MODELS_ROOT      │
        └────────────────────────────┘
```

---

## Validation Flowchart

```
User enters path: D:\WasteX\media

         │
         ▼
    ┌─────────────────────┐
    │ Path exists?        │
    └────┬────────────┬───┘
         │ NO         │ YES
         │            │
         ▼            ▼
    Create it     ┌──────────┐
         │        │Next check│
         │        └────┬─────┘
         │             │
         └──────┬──────┘
                │
                ▼
        ┌────────────────────┐
        │ Can write?         │
        │ (test write/delete)│
        └────┬────────────┬──┘
             │ NO         │ YES
             │            │
             ▼            ▼
        Permission  ┌──────────┐
        Error       │Next check│
                    └────┬─────┘
                         │
                         ▼
                    ┌──────────────────┐
                    │ Disk space free? │
                    │ (>= 50GB)        │
                    └────┬─────────┬───┘
                         │ NO      │ YES
                         │         │
                         ▼         ▼
                    Disk Full   ┌──────────┐
                    Error       │Save to DB│
                                └────┬─────┘
                                     │
                                     ▼
                                 ✅ SUCCESS
```

---

## Database Schema

```
┌────────────────────────────────────────────────────────────┐
│                   AppSettings Table                        │
├────────────────────────────────────────────────────────────┤
│ ID  │ KEY            │ VALUE             │ DESCRIPTION    │
├─────┼────────────────┼───────────────────┼────────────────┤
│  1  │ media_root     │ D:\WasteX\media   │ Uploaded...    │
│  2  │ datasets_root  │ D:\WasteX\datasets│ Training...    │
│  3  │ models_root    │ D:\WasteX\models  │ ML models      │
└────────────────────────────────────────────────────────────┘

Used by:
├─ Django settings.py: get_storage_path('media_root')
├─ File uploads: MEDIA_ROOT / 'uploads' / '2026' / '03' / '29'
├─ Training: DATASETS_ROOT / 'v1' / 'train' / 'Plastic'
└─ Models: MODELS_ROOT / 'logits_mdl.keras'
```

---

## File Organization

```
D:\WasteX\  (User-selected during installation)
│
├─ media/                          (MEDIA_ROOT)
│  └─ uploads/
│     └─ 2026/03/29/
│        ├─ image_001.jpg
│        ├─ image_002.jpg
│        └─ image_003.png
│
├─ datasets/                       (DATASETS_ROOT)
│  ├─ v1/                          (Training version 1)
│  │  ├─ train/                    (80% of data)
│  │  │  ├─ Plastic/
│  │  │  │  ├─ 001.jpg
│  │  │  │  └─ ...
│  │  │  ├─ Glass/
│  │  │  ├─ Paper/
│  │  │  └─ ...
│  │  ├─ val/                      (10% of data)
│  │  └─ test/                     (10% of data)
│  │
│  └─ v2/                          (Training version 2)
│     ├─ train/
│     ├─ val/
│     └─ test/
│
└─ models/                         (MODELS_ROOT)
   ├─ logits_mdl.keras
   ├─ classes.txt
   └─ versions/
      ├─ v1_20260224_162046/
      │  ├─ model.keras
      │  ├─ config.json
      │  └─ metrics.json
      └─ v1_20260225_123141/
         └─ ...
```

---

## Code Flow

```
Application Startup
│
├─ Django loads wastex/settings.py
│  │
│  ├─ Calls: MEDIA_ROOT = get_storage_path('WASTE_MEDIA_ROOT', 'media_root', 'media')
│  │
│  ├─ get_storage_path() function:
│  │  ├─ Try: os.getenv('WASTE_MEDIA_ROOT')  [For development]
│  │  ├─ Try: AppSettings.get('media_root')  [Production - from DB]
│  │  └─ Default: BASE_DIR / 'media'        [Fallback]
│  │
│  ├─ Returns: PosixPath('D:/WasteX/media')
│  │
│  └─ Same for DATASETS_ROOT and MODELS_ROOT
│
└─ App is ready to use configured paths

Runtime: Image Upload
│
├─ User uploads image
├─ View calls: default_storage.save('uploads/...', file)
├─ Saves to: MEDIA_ROOT / 'uploads' / '2026' / '03' / '29' / 'image.jpg'
│            ↑ This is from AppSettings!
├─ Gets logits from model
├─ Saves Image record to database
└─ Returns result to user

Runtime: Training
│
├─ User starts training
├─ Load dataset from: DATASETS_ROOT / 'v1' / 'train'
│                     ↑ This is from AppSettings!
├─ Train model
├─ Evaluate on test set from: DATASETS_ROOT / 'v1' / 'test'
├─ Save results to: MODELS_ROOT / 'versions' / 'v1_20260225_...'
└─ Database updated with results
```

---

## Paths Dictionary

```python
# In Django application:

# Configuration sources (in priority order):
SOURCES = {
    1: "Environment Variable",      # WASTE_MEDIA_ROOT=...
    2: "Database AppSettings",       # Saved during installation
    3: "Default folder",             # BASE_DIR / 'media'
}

# Actual paths after resolution:
PATHS = {
    'media_root':    'D:/WasteX/media',       # Images
    'datasets_root': 'D:/WasteX/datasets',    # Training data
    'models_root':   'D:/WasteX/models',      # ML models
}

# Used throughout app:
save_image_to = MEDIA_ROOT / 'uploads' / '2026/03/29'
load_train_from = DATASETS_ROOT / 'v1' / 'train' / 'Plastic'
save_model_to = MODELS_ROOT / 'versions' / 'v1_20260225_...'
```

---

## State Diagram

```
          ┌─────────────────┐
          │   Fresh Install │
          │  (No DB data)   │
          └────────┬────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  initialize_paths    │
        │  command runs        │
        │                      │
        │  Validates paths     │
        │  Saves to DB         │
        └────────┬─────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │   Configuration     │
        │   Set in Database   │
        │                     │
        │ AppSettings:        │
        │  media_root: ...    │
        │  datasets_root: ... │
        │  models_root: ...   │
        └────────┬────────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │  App Starts          │
        │  (Django runserver)  │
        │                      │
        │  Loads settings      │
        │  Queries AppSettings │
        │  Gets configured...  │
        │  paths from DB       │
        └────────┬─────────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │  Ready to Use        │
        │                      │
        │  • Upload images     │
        │  • Train models      │
        │  • View dashboard    │
        │  • etc.              │
        └──────────────────────┘
```

---

## Error Handling

```
User Input: D:\WasteX\media

         │
         ▼
    ┌──────────────────────┐
    │  Validation Check 1: │
    │  Path Exists?        │
    └────┬─────────┬───────┘
         │ NO      │ YES
         │         │
         ▼         ▼
    Create it    ┌────────┐
    Continue     │Check 2 │
    ✓            └────┬───┘
                      │
                      ▼
         ┌──────────────────────────┐
         │  Validation Check 2:     │
         │  Write Permission?       │
         │  (test write/delete)     │
         └────┬──────────┬──────────┘
              │ ERROR    │ OK
              │          │
              ▼          ▼
         ❌ Show     ┌──────────┐
         "Permission │ Check 3  │
         Denied"    └────┬─────┘
         Ask retry       │
                         ▼
         ┌──────────────────────────┐
         │  Validation Check 3:     │
         │  Disk Space >= 50GB?     │
         └────┬──────────┬──────────┘
              │ ERROR    │ OK
              │          │
              ▼          ▼
         ❌ Show     ✅ Save to DB
         "Low Space" Success!
         Ask retry
```

---

## Timeline

```
Installation Timeline:

00:00 - User runs install.bat
00:05 - Script checks Python
00:10 - Script asks for folder
00:15 - User enters: D:\WasteX
00:20 - Script validates path
00:45 - Script installs dependencies (pip install)
01:15 - Script runs migrations (django migrate)
01:20 - Script runs initialization (initialize_paths)
01:25 - Paths saved to database ✓
01:30 - Server starts
01:35 - Browser opens to http://localhost:8000/
01:40 - Ready to use! 🎉

Total time: ~1.5-2 minutes (first time, with downloads)
Subsequent runs: ~30 seconds
```

---

## Technology Stack

```
┌──────────────────────────────────────────┐
│  Installation-Time Configuration Stack   │
├──────────────────────────────────────────┤
│                                          │
│  Presentation:  install.bat (Windows)   │
│                 Batch script             │
│                 User-friendly prompts    │
│                 Validates and summarizes │
│                                          │
│  Application:   initialize_paths command │
│                 Django management command│
│                 Comprehensive validation │
│                 Pretty console output    │
│                                          │
│  Storage:       AppSettings model        │
│                 Django ORM               │
│                 SQLite database          │
│                 Persistent configuration │
│                                          │
│  Runtime:       settings.py function     │
│                 get_storage_path()       │
│                 Fallback logic           │
│                 Environment variable compat │
│                                          │
│  Files:         MEDIA_ROOT               │
│                 DATASETS_ROOT            │
│                 MODELS_ROOT              │
│                 OS file system           │
│                                          │
└──────────────────────────────────────────┘
```

---

## Comparison Matrix

```
┌─────────────────┬──────────────┬──────────────┬────────────────┐
│ Aspect          │ Before (.env)│ After (AppS) │ Improvement    │
├─────────────────┼──────────────┼──────────────┼────────────────┤
│ User Interface  │ Edit text    │ install.bat  │ User-friendly  │
│                 │ (technical)  │ (automated)  │                │
├─────────────────┼──────────────┼──────────────┼────────────────┤
│ Validation      │ None         │ Comprehensive│ Reliable       │
│                 │              │              │                │
├─────────────────┼──────────────┼──────────────┼────────────────┤
│ Error Messages  │ Missing      │ Clear + Tips │ Professional   │
│                 │              │              │                │
├─────────────────┼──────────────┼──────────────┼────────────────┤
│ Persistence     │ Manual edit  │ Database     │ Automatic      │
│                 │ (error-prone)│ (reliable)   │                │
├─────────────────┼──────────────┼──────────────┼────────────────┤
│ Query/Debug     │ grep .env    │ ORM query    │ Programmable   │
│                 │              │              │                │
├─────────────────┼──────────────┼──────────────┼────────────────┤
│ Professional    │ Unprofessional│ Enterprise  │ Credible       │
│ Image          │              │ quality      │                │
└─────────────────┴──────────────┴──────────────┴────────────────┘
```

---

**This visual guide helps explain the installation system quickly and clearly.** ✓
