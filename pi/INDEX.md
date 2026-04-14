# Pi Folder Structure - Complete Organization

Everything related to Raspberry Pi image processing is now organized in a single `pi/` folder.

## Folder Structure

```
pi/
├── README.md                      ← Start here
├── requirements.txt               ← Python dependencies
│
├── scripts/
│   └── image_watcher.py          ← Main watcher script
│
├── systemd/
│   └── pi-image-watcher.service  ← Auto-start service
│
└── docs/
    ├── SETUP.md                  ← Installation guide
    ├── TROUBLESHOOTING.md        ← Common issues
    └── API_REFERENCE.md          ← Backend API docs
```

## File Guide

### scripts/

**`image_watcher.py`**
- Main watcher script for Raspberry Pi
- Monitors folder for new images
- Uploads to backend automatically
- Includes retry logic and error handling
- ~400 lines of production code

**Installation**:
```powershell
scp pi/scripts/image_watcher.py dhruba001@192.169.0.120:~/
```

### systemd/

**`pi-image-watcher.service`**
- Systemd service file for auto-start on Pi boot
- Handles auto-restart on failure
- Proper logging configuration
- Ready to deploy

**Installation**:
```bash
sudo cp pi/systemd/pi-image-watcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pi-image-watcher
```

### docs/

**`SETUP.md`**
- Complete step-by-step setup guide
- Installation instructions
- Configuration guide
- Testing procedures
- Integration examples

**`TROUBLESHOOTING.md`**
- Common issues and solutions
- Debugging commands
- Network troubleshooting
- Service debugging
- Quick fix checklist

**`API_REFERENCE.md`**
- Backend API endpoints documentation
- Request/response examples
- Error handling
- Performance characteristics
- Testing examples

### Root Files

**`README.md`**
- Quick start guide
- Folder structure overview
- Key commands
- Documentation links

**`requirements.txt`**
- Python dependencies for Pi
- Install with: `pip install -r requirements.txt`

---

## Quick Start

### Copy to Pi

```powershell
# Copy everything
scp -r pi dhruba001@192.169.0.120:~/wastex_pi

# Or just the script
scp pi/scripts/image_watcher.py dhruba001@192.169.0.120:~/
```

### Install Dependencies

```bash
pip install -r pi/requirements.txt
# or
pip install watchdog requests pillow
```

### Configure

```bash
# Edit the script
nano ~/image_watcher.py

# Update these lines:
BACKEND_URL = "http://192.169.0.111:8000"
WATCH_FOLDER = "/home/dhruba001/webcam_captures"
SOURCE_ID = "pi_001"
```

### Run

```bash
python3 ~/image_watcher.py
```

---

## What's Different from Root

The old `pi_image_watcher.py` and related files in the root are now organized in:

```
BEFORE:
wastex/
├── pi_image_watcher.py
├── pi-image-watcher.service
├── PI_QUICK_START.md
├── PI_REALTIME_SETUP.md
└── ...

AFTER:
wastex/
└── pi/
    ├── scripts/image_watcher.py
    ├── systemd/pi-image-watcher.service
    ├── docs/SETUP.md
    ├── docs/TROUBLESHOOTING.md
    ├── docs/API_REFERENCE.md
    ├── README.md
    └── requirements.txt
```

---

## Backend Integration

The Pi folder works with your Django backend at:

```
Backend API Endpoints:
POST   /api/pi/inference/
POST   /api/pi/batch-inference/
GET    /api/pi/health/
```

See `classifier/views/pi_inference_api.py` in the main project.

---

## Organization Benefits

✅ **Organized** - All Pi code in one place
✅ **Clean** - Root folder not cluttered
✅ **Portable** - Easy to copy to Pi
✅ **Documented** - Complete guides included
✅ **Maintainable** - Clear structure
✅ **Scalable** - Easy to add new devices/scripts

---

## Next Steps

1. **Start with**: `pi/README.md`
2. **For setup**: `pi/docs/SETUP.md`
3. **For issues**: `pi/docs/TROUBLESHOOTING.md`
4. **For API**: `pi/docs/API_REFERENCE.md`

---

## Old Files

The following files in the root are now superseded by the `pi/` folder:

- `pi_image_watcher.py` → `pi/scripts/image_watcher.py`
- `pi-image-watcher.service` → `pi/systemd/pi-image-watcher.service`
- `PI_QUICK_START.md` → `pi/docs/SETUP.md`
- `PI_REALTIME_SETUP.md` → `pi/docs/SETUP.md`
- `PI_ARCHITECTURE_DIAGRAM.md` → Backend docs
- `PI_IMPLEMENTATION_SUMMARY.md` → Backend docs
- `PI_COMPLETE_DEPLOYMENT_GUIDE.md` → `pi/docs/SETUP.md`

You can delete the root-level PI_* files once you've reviewed the new organization.

---

## File Sizes

```
pi/scripts/image_watcher.py         ~12 KB  (400 lines)
pi/systemd/pi-image-watcher.service ~0.5 KB (20 lines)
pi/docs/SETUP.md                    ~8 KB   (250 lines)
pi/docs/TROUBLESHOOTING.md          ~12 KB  (400 lines)
pi/docs/API_REFERENCE.md            ~10 KB  (350 lines)
pi/README.md                        ~4 KB   (150 lines)
pi/requirements.txt                 ~0.2 KB (3 lines)

Total: ~47 KB
```

---

## Copying to Your Pi

### Option 1: Copy Entire Folder

```powershell
scp -r pi dhruba001@192.169.0.120:~/wastex_pi
```

Then on Pi:
```bash
cd ~/wastex_pi
pip install -r requirements.txt
# Edit scripts/image_watcher.py
python3 scripts/image_watcher.py
```

### Option 2: Copy Just Script

```powershell
scp pi/scripts/image_watcher.py dhruba001@192.169.0.120:~/
pip install watchdog requests pillow
python3 ~/image_watcher.py
```

### Option 3: Copy and Install Service

```powershell
scp -r pi dhruba001@192.169.0.120:~/wastex_pi
```

Then on Pi:
```bash
sudo cp ~/wastex_pi/systemd/pi-image-watcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pi-image-watcher
sudo systemctl start pi-image-watcher
```

---

## Configuration Template

Edit `pi/scripts/image_watcher.py`:

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION - EDIT THESE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WATCH_FOLDER = "/home/dhruba001/webcam_captures"  # Where images are saved
BACKEND_URL = "http://192.169.0.111:8000"         # Backend IP:port
SOURCE_ID = "pi_001"                               # This Pi's ID
```

That's it! Everything else is automatic.

---

## Summary

All Raspberry Pi-related code and documentation is now in the `pi/` folder:

- **Scripts** - Ready-to-run Python code
- **Systemd** - Service files for auto-start
- **Docs** - Complete setup and troubleshooting guides
- **Requirements** - Python dependencies

The entire folder can be copied to your Raspberry Pi for easy deployment!

---

See `pi/README.md` to get started! 🚀
