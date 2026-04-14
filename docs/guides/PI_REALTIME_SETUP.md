# Raspberry Pi Real-Time Image Processing Setup Guide

## Overview

This setup allows you to:
1. Capture images on Raspberry Pi via webcam
2. Automatically upload them to your backend
3. Run inference immediately
4. Get results back to the Pi

## Architecture

```
Raspberry Pi Webcam Script
        ↓ (saves to folder)
Watch Folder (/home/dhruba001/webcam_captures)
        ↓ (detects new files)
pi_image_watcher.py (monitors folder)
        ↓ (uploads via HTTP)
Backend Server (receives & processes)
        ↓ (runs inference)
Results (returned to Pi)
```

## Installation Steps

### 1. Create Capture Folder on Raspberry Pi

```bash
mkdir -p ~/webcam_captures
```

### 2. Copy the Watcher Script to Pi

From your local machine:

```powershell
# Copy the script to your Pi
scp C:\WASTE\wastex\pi_image_watcher.py dhruba001@192.169.0.111:~/pi_image_watcher.py
```

### 3. Install Dependencies on Raspberry Pi

```bash
# SSH into Pi
ssh dhruba001@192.169.0.111

# Install required packages
pip install watchdog requests

# Make script executable
chmod +x ~/pi_image_watcher.py
```

### 4. Configure the Script

Edit `~/pi_image_watcher.py` on the Pi:

```bash
nano ~/pi_image_watcher.py
```

Update these variables:

```python
# Point to your capture folder
WATCH_FOLDER = "/home/dhruba001/webcam_captures"

# Your backend server IP (get this from your laptop)
BACKEND_URL = "http://192.168.1.100:8000"  # ← Update this!

# Unique ID for this Pi
SOURCE_ID = "pi_001"
```

To find your backend IP on Windows:
```powershell
ipconfig
```

Look for "IPv4 Address" under your active network connection.

### 5. Test the Setup

#### Step A: Start the Watcher on Pi

```bash
python3 ~/pi_image_watcher.py
```

You should see:
```
✅ Backend is online and ready!
👀 Watching for new images...
```

#### Step B: Start Your Backend

On your local machine:

```powershell
cd C:\WASTE\wastex
python manage.py runserver 0.0.0.0:8000
```

#### Step C: Create a Test Image

On the Pi, create a test image in the watch folder:

```bash
# Create a simple test image
python3 << 'EOF'
from PIL import Image
img = Image.new('RGB', (100, 100), color='red')
img.save('/home/dhruba001/webcam_captures/test_image.jpg')
EOF
```

#### Step D: Check Results

In the watcher output on Pi, you should see:
```
📷 New image detected: /home/dhruba001/webcam_captures/test_image.jpg
✅ Upload successful!
   Predictions: {'class_name': 'plastic', 'confidence': 0.95}
   Image ID: 123
```

## Running Watcher Automatically (Systemd Service)

To run the watcher automatically on Pi startup:

### 1. Create Service File

```bash
sudo nano /etc/systemd/system/pi-image-watcher.service
```

### 2. Copy Content

```ini
[Unit]
Description=Raspberry Pi Real-time Image Watcher
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=dhruba001
WorkingDirectory=/home/dhruba001
ExecStart=/usr/bin/python3 /home/dhruba001/pi_image_watcher.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 3. Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable pi-image-watcher
sudo systemctl start pi-image-watcher

# Check status
sudo systemctl status pi-image-watcher

# View logs
sudo journalctl -u pi-image-watcher -f
```

## Troubleshooting

### Issue: "Backend is offline"

**Solution:** Make sure:
- Your backend is running: `python manage.py runserver 0.0.0.0:8000`
- Firewall is not blocking port 8000
- Pi can reach your backend IP: `ping 192.168.1.100`
- BACKEND_URL is correct in the script

### Issue: "Connection refused"

**Solution:** Check:
- Backend IP is correct
- Backend is running on port 8000
- Pi has internet connectivity

### Issue: Images aren't being uploaded

**Solution:**
1. Check that images are being saved to the watch folder
2. Check watcher logs: `tail -f /tmp/pi_image_watcher_pi_001.log`
3. Ensure folder permissions: `chmod 755 ~/webcam_captures`

### Issue: "No module named watchdog"

**Solution:**
```bash
pip install --upgrade watchdog requests
```

## Integration with Your Webcam Script

Your webcam script should save images here:
```python
import os
CAPTURE_FOLDER = "/home/dhruba001/webcam_captures"
output_path = os.path.join(CAPTURE_FOLDER, f"capture_{timestamp}.jpg")
```

The watcher will automatically detect them and upload to your backend!

## API Endpoints

Your backend now has these new endpoints:

### Single Image Inference
```
POST /api/pi/inference/
Content-Type: multipart/form-data

Parameters:
- image: image file
- source: device identifier (e.g., "pi_001")

Response:
{
  "status": "success",
  "predictions": {
    "class_name": "plastic",
    "confidence": 0.95
  },
  "image_id": 123,
  "timestamp": "2026-04-13T18:51:52.123456Z"
}
```

### Batch Image Inference
```
POST /api/pi/batch-inference/
Content-Type: multipart/form-data

Parameters:
- images: list of image files
- source: device identifier

Response:
{
  "status": "success",
  "total_processed": 5,
  "results": [...]
}
```

### Health Check
```
GET /api/pi/health/

Response:
{
  "status": "online",
  "timestamp": "2026-04-13T18:51:52.123456Z"
}
```

## Next Steps

1. ✅ Set up the watcher script
2. ✅ Configure BACKEND_URL
3. ✅ Test with a sample image
4. ✅ Integrate with your webcam script
5. ✅ Set up systemd service for auto-start

Once everything works, images will flow seamlessly from Pi to backend!

## Monitoring Dashboard

To monitor uploads in real-time:

```bash
# On your local machine
tail -f \\\\192.169.0.111\\home\\dhruba001\\pi_image_watcher_pi_001.log
```

Or check your backend logs for received images:

```powershell
# On your backend server
python manage.py dbshell
SELECT * FROM classifier_image ORDER BY created_at DESC LIMIT 10;
```
