# Raspberry Pi Setup Guide

## ⚠️ Important: IP Address Setup

Since both are on the **same WiFi network**, here's the setup:

| Device | IP Address | Purpose |
|--------|-----------|---------|
| **Windows (Your PC)** | `192.169.0.111` | Runs Django backend at http://127.0.0.1:8000 |
| **Raspberry Pi** | `192.169.0.120` | On same WiFi, runs image watcher |
| **Django Backend** | `http://127.0.0.1:8000` on Windows | Only localhost on Windows machine |
| **Pi connects to backend** | `http://192.169.0.111:8000` | Pi uses Windows' WiFi IP address! |

**The Setup**: Both on same WiFi (192.169.0.x network). Pi reaches your Django server using Windows IP `192.169.0.111:8000`

## Prerequisites

- Raspberry Pi with Python 3.7+
- Network connectivity to your backend
- Camera connected to Pi
- Windows machine's IP address (run `ipconfig` on Windows first!)

## Installation Steps

### Step 1: Copy Script to Pi

From your Windows machine:

```powershell
cd C:\WASTE\wastex
scp pi/scripts/image_watcher.py dhruba001@192.169.0.120:~/
```

### Step 2: SSH into Pi

```bash
ssh dhruba001@192.169.0.120
```

### Step 3: Create Virtual Environment

```bash
python3 -m venv ~/wastex_venv
source ~/wastex_venv/bin/activate
```

### Step 4: Create Capture Folder

```bash
mkdir -p ~/webcam_captures
chmod 755 ~/webcam_captures
```

### Step 5: Install Dependencies

```bash
pip install --upgrade pip
pip install watchdog requests pillow
```

### Step 6: Configure the Script

Your Windows machine's IP is: **192.169.0.111**

Edit the Pi script:

```bash
nano ~/image_watcher.py
```

Update these variables (use your Windows IP: `192.169.0.111`):

```python
# Your Windows machine's WiFi IP
BACKEND_URL = "http://192.169.0.111:8000"

# Location of capture folder
WATCH_FOLDER = "/home/dhruba001/webcam_captures"

# Device identifier (for multi-Pi setups)
SOURCE_ID = "pi_001"
```

**Steps to edit in nano**:
1. Press `Ctrl+W` to search for "BACKEND_URL"
2. Change `127.0.0.1` to `192.169.0.111`
3. Press `Ctrl+O` then `Enter` to save
4. Press `Ctrl+X` to exit

### Step 7: Test the Watcher

Make sure your virtual environment is activated:

```bash
source ~/wastex_venv/bin/activate
```

Then run the watcher:

```b
```

You should see:
```
======================================================================
🚀 Raspberry Pi Image Watcher Service Started
======================================================================
✅ Backend is online and ready!
📁 Watching folder: /home/dhruba001/webcam_captures
🔌 Backend URL: http://192.169.0.111:8000
🎯 Source ID: pi_001
👀 Watching for new images...
```

### Step 8: Test with a Sample Image

In another terminal on the Pi:

```bash
python3 << 'EOF'
from PIL import Image
img = Image.new('RGB', (640, 480), color='red')
img.save('/home/dhruba001/webcam_captures/test.jpg')
EOF
```

In the watcher output, you should see:

```
📷 New image detected: /home/dhruba001/webcam_captures/test.jpg
✅ Upload successful!
   Predictions: {'class_name': 'plastic', 'confidence': 0.95}
   Image ID: 123
```

## Auto-Start on Boot (Optional)

### Step 1: Copy Service File

From your local machine:

```powershell
scp pi\systemd\pi-image-watcher.service dhruba001@192.169.0.120:~/
```

### Step 2: Install Service

On the Pi:

```bash
sudo cp ~/pi-image-watcher.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### Step 3: Enable Service

```bash
sudo systemctl enable pi-image-watcher
sudo systemctl start pi-image-watcher
```

### Step 4: Verify

```bash
sudo systemctl status pi-image-watcher
```

### Step 5: View Logs

```bash
sudo journalctl -u pi-image-watcher -f
```

## Integration with Your Webcam Script

Your webcam script should save images to the watch folder:

```python
import os
from datetime import datetime

CAPTURE_FOLDER = "/home/dhruba001/webcam_captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

def capture_frame(frame):
    """Save frame to capture folder."""
    timestamp = datetime.now().isoformat()
    output_path = os.path.join(CAPTURE_FOLDER, f"frame_{timestamp}.jpg")
    cv2.imwrite(output_path, frame)
    # The watcher will automatically upload!
```

That's it! The watcher handles the rest.

## Monitoring

### View Watcher Logs

```bash
# Live stream
tail -f /tmp/pi_image_watcher_pi_001.log

# Last 50 lines
tail -50 /tmp/pi_image_watcher_pi_001.log

# Search for errors
grep "❌" /tmp/pi_image_watcher_pi_001.log
```

### Check Upload Status

```bash
# Count uploaded images
grep "✅ Upload successful" /tmp/pi_image_watcher_pi_001.log | wc -l

# Check failed uploads
grep "❌" /tmp/pi_image_watcher_pi_001.log
```

## Next Steps

1. ✅ Script is running
2. ✅ Images are uploading
3. ✅ Integrate with your webcam script
4. ✅ Set up systemd service (optional)
5. ✅ Configure monitoring/alerts

For troubleshooting, see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**.
