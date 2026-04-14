# Complete Pi Real-Time Inference Setup & Deployment Guide

> **Goal**: Images captured on Raspberry Pi are automatically uploaded to your backend for instant inference.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Detailed Setup](#detailed-setup)
3. [Testing & Validation](#testing--validation)
4. [Troubleshooting](#troubleshooting)
5. [Production Deployment](#production-deployment)

---

## Quick Start

### Prerequisites

- Raspberry Pi (any model) with Python 3.7+
- Your backend running on a laptop/server with Django
- Network connectivity between Pi and backend
- Camera connected to Pi

### Step 1: Copy Script to Pi

```powershell
# On your Windows machine
scp C:\WASTE\wastex\pi_image_watcher.py dhruba001@192.169.0.111:~/
```

### Step 2: Install Dependencies on Pi

```bash
ssh dhruba001@192.169.0.111
pip install watchdog requests pillow
mkdir -p ~/webcam_captures
```

### Step 3: Configure Script

```bash
nano ~/pi_image_watcher.py
```

Update these lines:
```python
BACKEND_URL = "http://192.168.1.100:8000"  # Your PC IP here
WATCH_FOLDER = "/home/dhruba001/webcam_captures"
SOURCE_ID = "pi_001"
```

**To find your PC IP:**
```powershell
ipconfig  # Look for IPv4 Address (e.g., 192.168.1.100)
```

### Step 4: Start Backend

```powershell
cd C:\WASTE\wastex
python manage.py migrate  # Apply new migration
python manage.py runserver 0.0.0.0:8000
```

### Step 5: Start Watcher on Pi

```bash
python3 ~/pi_image_watcher.py
```

Expected output:
```
✅ Backend is online and ready!
📁 Watching folder: /home/dhruba001/webcam_captures
🔌 Backend URL: http://192.168.1.100:8000
👀 Watching for new images...
```

### Step 6: Test

Create a test image on Pi:
```bash
python3 << 'EOF'
from PIL import Image
img = Image.new('RGB', (640, 480), color='red')
img.save('/home/dhruba001/webcam_captures/test.jpg')
EOF
```

Watch for success message:
```
📷 New image detected: /home/dhruba001/webcam_captures/test.jpg
✅ Upload successful!
   Predictions: {'class_name': 'plastic', 'confidence': 0.95}
   Image ID: 123
```

**That's it!** You're ready to go. 🎉

---

## Detailed Setup

### Architecture Overview

```
Webcam → Pi Folder → Watcher Script → Backend API → Inference → Database
```

### Component Details

#### 1. Raspberry Pi File Watcher

**File**: `pi_image_watcher.py`

**What it does**:
- Monitors a folder 24/7
- Detects new image files
- Uploads to backend immediately
- Retries on failure
- Full logging to `/tmp/pi_image_watcher_*.log`

**Requirements**:
```bash
pip install watchdog requests pillow
```

**Configuration variables** (in the script):

```python
WATCH_FOLDER = "/home/dhruba001/webcam_captures"  # Where your camera saves
BACKEND_URL = "http://192.168.1.100:8000"         # Your backend server
SOURCE_ID = "pi_001"                               # Device identifier
MAX_RETRIES = 3                                    # Upload retry attempts
RETRY_DELAY = 2                                    # Retry wait in seconds
```

#### 2. Backend API Endpoints

**Endpoints added to your Django backend**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/pi/inference/` | POST | Upload single image for inference |
| `/api/pi/batch-inference/` | POST | Upload multiple images |
| `/api/pi/health/` | GET | Check if backend is online |

**Endpoint Details**:

##### Single Image Inference

```http
POST /api/pi/inference/
Content-Type: multipart/form-data

image: <binary image file>
source: pi_001
```

Response:
```json
{
  "status": "success",
  "predictions": {
    "class_name": "plastic",
    "confidence": 0.9823
  },
  "image_id": 12345,
  "timestamp": "2026-04-13T18:51:52.123456Z",
  "source": "pi_001"
}
```

##### Batch Image Inference

```http
POST /api/pi/batch-inference/
Content-Type: multipart/form-data

images: <file1>, <file2>, <file3>
source: pi_001
```

Response:
```json
{
  "status": "success",
  "total_processed": 3,
  "results": [
    {"image_id": 123, "predictions": {...}},
    {"image_id": 124, "predictions": {...}},
    {"image_id": 125, "predictions": {...}}
  ],
  "source": "pi_001"
}
```

#### 3. Database Integration

New fields added to `Image` model:

```python
source_device = CharField(max_length=100)  # e.g., "pi_001"
predicted_label = CharField(max_length=100) # Predicted class
confidence = FloatField()                   # Confidence score
```

Query recent Pi uploads:
```python
from classifier.models import Image

# Last 10 images from pi_001
recent = Image.objects.filter(
    source_device='pi_001'
).order_by('-uploaded_at')[:10]
```

---

## Testing & Validation

### Test 1: Health Check

```bash
# On Pi
curl http://192.168.1.100:8000/api/pi/health/
```

Expected response:
```json
{
  "status": "online",
  "timestamp": "2026-04-13T18:51:52.123456Z",
  "version": "1.0"
}
```

### Test 2: Single Image Upload

```bash
# On Pi
curl -X POST \
  -F "image=@/home/dhruba001/webcam_captures/test.jpg" \
  -F "source=pi_001" \
  http://192.168.1.100:8000/api/pi/inference/
```

### Test 3: Batch Upload

```bash
# On Pi
curl -X POST \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "source=pi_001" \
  http://192.168.1.100:8000/api/pi/batch-inference/
```

### Test 4: Automated Test Script

```bash
# On local machine
cd C:\WASTE\wastex
python test_pi_api.py http://192.168.1.100:8000 sample_image.jpg
```

### Test 5: End-to-End Flow

**On Pi:**
```bash
# Terminal 1 - Watch the watcher
tail -f /tmp/pi_image_watcher_pi_001.log

# Terminal 2 - Create test images
python3 << 'EOF'
from PIL import Image
from datetime import datetime
import time

for i in range(5):
    img = Image.new('RGB', (640, 480), color='green')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img.save(f'/home/dhruba001/webcam_captures/test_{i}_{timestamp}.jpg')
    time.sleep(2)  # 2 second interval
EOF
```

Expected in logs:
```
📷 New image detected: /home/dhruba001/webcam_captures/test_0_...jpg
✅ Upload successful!
   Predictions: {'class_name': 'plastic', 'confidence': 0.88}
   Image ID: 100
📷 New image detected: /home/dhruba001/webcam_captures/test_1_...jpg
✅ Upload successful!
   Predictions: {'class_name': 'paper', 'confidence': 0.92}
   Image ID: 101
...
```

---

## Troubleshooting

### Problem: "Backend is offline"

**Diagnosis**:
```bash
# On Pi, check if you can reach backend
ping 192.168.1.100
curl http://192.168.1.100:8000/api/pi/health/
```

**Solutions**:
1. Make sure backend is running: `python manage.py runserver 0.0.0.0:8000`
2. Check BACKEND_URL in script is correct
3. Check firewall is not blocking port 8000
4. Verify Pi has internet connectivity

### Problem: "Connection refused"

**Diagnosis**:
```bash
# Check if backend is listening
netstat -an | grep 8000  # Windows
ss -an | grep 8000       # Linux
```

**Solutions**:
1. Start backend: `python manage.py runserver 0.0.0.0:8000`
2. Check port 8000 is not in use: `lsof -i :8000`
3. Try different port if needed (update in script)

### Problem: Images detected but not uploaded

**Diagnosis**:
```bash
# Check watcher logs
tail -50 /tmp/pi_image_watcher_pi_001.log

# Check if images are being created
ls -la ~/webcam_captures/ | tail -20
```

**Solutions**:
1. Ensure folder has correct permissions: `chmod 755 ~/webcam_captures`
2. Verify BACKEND_URL is accessible: `curl -v http://backend:8000/api/pi/health/`
3. Check network connectivity: `ping 8.8.8.8`
4. Restart watcher: Kill process and run again

### Problem: "No module named watchdog"

**Solution**:
```bash
pip install --upgrade watchdog requests pillow
```

### Problem: Images uploaded but no inference results

**Diagnosis**:
```bash
# Check backend logs
tail -f /var/log/django/production.log  # if using production

# Query database for images
python manage.py shell
>>> from classifier.models import Image
>>> Image.objects.filter(source_device='pi_001').count()
10
>>> Image.objects.filter(source_device='pi_001').last().predicted_label
'plastic'
```

**Solutions**:
1. Verify model file exists: `ls models/logits_mdl.keras`
2. Check Django migrations applied: `python manage.py migrate`
3. Verify TensorFlow installed: `pip install tensorflow`
4. Check backend error logs

### Problem: Model loading is slow

**Info**: First inference is slower as model is loaded into RAM

**Solution**: Model stays loaded, subsequent inferences are fast (~500ms)

### Problem: USB connection drops

**Solution on Pi**:
```bash
# Keep USB connection alive
sudo sh -c 'echo "options usbcore autosuspend=-1" > /etc/modprobe.d/usb_suspend.conf'
sudo reboot
```

---

## Production Deployment

### Option 1: Systemd Service (Recommended)

**On Pi**:

```bash
# Copy service file
sudo cp /home/dhruba001/pi-image-watcher.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable pi-image-watcher
sudo systemctl start pi-image-watcher

# Verify
sudo systemctl status pi-image-watcher

# View logs
sudo journalctl -u pi-image-watcher -f
```

**Service file** (`/etc/systemd/system/pi-image-watcher.service`):

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

### Option 2: Cron Job

```bash
# Add to crontab
crontab -e

# Add this line:
@reboot python3 /home/dhruba001/pi_image_watcher.py >> /tmp/pi_watcher.log 2>&1
```

### Option 3: Docker Container (Advanced)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY pi_image_watcher.py .

RUN pip install watchdog requests pillow

CMD ["python3", "pi_image_watcher.py"]
```

```bash
docker build -t pi-watcher .
docker run -d -v /home/dhruba001/webcam_captures:/images \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  pi-watcher
```

### Monitoring in Production

**Check service status**:
```bash
sudo systemctl status pi-image-watcher
```

**View recent logs**:
```bash
sudo journalctl -u pi-image-watcher -n 50
```

**Monitor uploads**:
```bash
# On backend machine
python manage.py shell
>>> from classifier.models import Image
>>> Image.objects.filter(source_device='pi_001').count()
>>> Image.objects.filter(source_device='pi_001').aggregate(Avg('confidence'))
```

**Dashboard query**:
```python
from django.db.models import Count, Avg
from classifier.models import Image

# Get stats for Pi uploads
stats = Image.objects.filter(
    source_device='pi_001'
).aggregate(
    total=Count('id'),
    avg_confidence=Avg('confidence'),
    latest=Max('created_at')
)
```

---

## Integration with Your Webcam Script

Your webcam capture script should save images here:

```python
import os
import cv2
from datetime import datetime

# Configuration
CAPTURE_FOLDER = "/home/dhruba001/webcam_captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# Capture loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if ret:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = os.path.join(CAPTURE_FOLDER, f"frame_{timestamp}.jpg")
        
        # Save frame (watcher will automatically upload)
        cv2.imwrite(filename, frame)
        
        # Display or process frame
        cv2.imshow('Capture', frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**That's all!** The watcher automatically detects and uploads images.

---

## Performance Optimization

### For High-Frequency Uploads

If capturing many images per second:

1. **Batch upload**:
```python
# Collect 5 images before uploading
MAX_BATCH_SIZE = 5
batch = []

for image_path in watch_folder:
    batch.append(image_path)
    if len(batch) >= MAX_BATCH_SIZE:
        upload_batch(batch)
        batch = []
```

2. **Adjust retry settings**:
```python
MAX_RETRIES = 1  # Reduce retries for speed
RETRY_DELAY = 0.5  # Shorter delay
```

3. **Use compression**:
```python
img.save(path, 'JPEG', quality=85)  # Lower quality = smaller file
```

### For Low-Latency Requirements

1. Keep model in memory (it is by default)
2. Use lighter model if possible
3. Reduce image resolution on Pi before upload
4. Use HTTP/2 for connection reuse

---

## Summary

✅ **Files created**:
- `pi_image_watcher.py` - Main watcher script
- `classifier/views/pi_inference_api.py` - Backend endpoints
- `classifier/migrations/0012_image_source_device.py` - Database migration
- `pi-image-watcher.service` - Systemd service file
- `test_pi_api.py` - API testing script

✅ **Database changes**:
- Added `source_device` field to Image model
- Added `predicted_label` field to Image model
- Created migration file

✅ **URLs updated**:
- `/api/pi/inference/` - Single image
- `/api/pi/batch-inference/` - Multiple images
- `/api/pi/health/` - Health check

✅ **Ready to deploy!**

---

## Quick Reference

```bash
# On Pi - Install
pip install watchdog requests pillow

# On Pi - Run
python3 ~/pi_image_watcher.py

# On Pi - Enable on startup
sudo systemctl enable pi-image-watcher

# On Backend - Apply migration
python manage.py migrate

# On Backend - Start server
python manage.py runserver 0.0.0.0:8000

# On Backend - Check uploads
python manage.py shell
>>> from classifier.models import Image
>>> Image.objects.filter(source_device='pi_001').count()
```

That's it! 🚀
