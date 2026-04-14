# Backend API Reference

The image watcher script communicates with your backend using these API endpoints.

## Base URL

```
http://192.169.0.111:8000
```

This is your Windows backend machine's IP on the WiFi network.
The backend is running Django development server.

---

## Endpoints

### 1. Single Image Inference

Upload a single image for inference.

```http
POST /api/pi/inference/
Content-Type: multipart/form-data
```

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file | Yes | Image file to process |
| `source` | string | Yes | Device ID (e.g., "pi_001") |

**Example**:

```bash
curl -X POST \
  -F "image=@photo.jpg" \
  -F "source=pi_001" \
  http://192.169.0.111:8000/api/pi/inference/
```

**Response** (200 OK):

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

**Error Response** (400/500):

```json
{
  "status": "error",
  "message": "Error description"
}
```

---

### 2. Batch Image Inference

Upload multiple images at once.

```http
POST /api/pi/batch-inference/
Content-Type: multipart/form-data
```

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `images` | file[] | Yes | Multiple image files |
| `source` | string | Yes | Device ID |

**Example**:

```bash
curl -X POST \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg" \
  -F "source=pi_001" \
  http://192.169.0.111:8000/api/pi/batch-inference/
```

**Response** (200 OK):

```json
{
  "status": "success",
  "total_processed": 3,
  "results": [
    {
      "image_id": 100,
      "predictions": {
        "class_name": "plastic",
        "confidence": 0.95
      },
      "filename": "image1.jpg",
      "timestamp": "2026-04-13T18:51:52Z"
    },
    {
      "image_id": 101,
      "predictions": {
        "class_name": "paper",
        "confidence": 0.92
      },
      "filename": "image2.jpg",
      "timestamp": "2026-04-13T18:51:53Z"
    },
    {
      "image_id": 102,
      "predictions": {
        "class_name": "metal",
        "confidence": 0.88
      },
      "filename": "image3.jpg",
      "timestamp": "2026-04-13T18:51:54Z"
    }
  ],
  "source": "pi_001"
}
```

---

### 3. Health Check

Check if the backend is online and ready.

```http
GET /api/pi/health/
```

**Example**:

```bash
curl http://192.169.0.111:8000/api/pi/health/
```

**Response** (200 OK):

```json
{
  "status": "online",
  "timestamp": "2026-04-13T18:51:52.123456Z",
  "version": "1.0"
}
```

**Error Response** (connection refused):

The watcher will retry uploads when the backend comes online.

---

## Response Fields

### Predictions Object

```json
{
  "class_name": "plastic",    // Waste classification
  "confidence": 0.9823         // 0.0 to 1.0
}
```

**Possible Classes**:
- `plastic`
- `metal`
- `glass`
- `paper`
- `organic`

### Error Messages

| Message | Meaning | Solution |
|---------|---------|----------|
| No image provided | POST data missing `image` field | Add file upload |
| Backend error | Server returned non-200 status | Check backend logs |
| Connection refused | Can't reach backend | Check IP and port |
| Model not found | TensorFlow model missing | Check model file exists |

---

## Testing

### Using cURL

```bash
# Single image
curl -X POST \
  -F "image=@test.jpg" \
  -F "source=test_pi" \
  http://localhost:8000/api/pi/inference/

# Health check
curl http://localhost:8000/api/pi/health/
```

### Using Python

```python
import requests

# Single image
with open('test.jpg', 'rb') as f:
    files = {'image': f}
    data = {'source': 'pi_001'}
    response = requests.post(
        'http://192.169.0.111:8000/api/pi/inference/',
        files=files,
        data=data
    )
    print(response.json())

# Health check
response = requests.get('http://192.169.0.111:8000/api/pi/health/')
print(response.json())
```

### Using Test Script

```powershell
cd C:\WASTE\wastex
python test_pi_api.py http://192.169.0.111:8000 sample.jpg
```

---

## Error Handling

### Connection Errors

The watcher automatically retries 3 times with exponential backoff:

```
Attempt 1: Wait 2 seconds
Attempt 2: Wait 2 seconds
Attempt 3: Wait 2 seconds
Failed: Log error and continue
```

### Timeout

Default timeout: **30 seconds** per request

To change, edit `image_watcher.py`:

```python
timeout=60  # Increase to 60 seconds
```

---

## Performance

### Expected Timing

```
Single Image:
├─ Network upload: 100-200ms
├─ Server processing: 1200-1800ms
└─ Total: ~2 seconds

Batch (5 images):
├─ Upload: 500ms
├─ Processing: ~1500ms (model loaded once)
└─ Total: ~2 seconds for 5 images
```

### Rate Limiting

No rate limiting by default. For production, consider implementing per-device limits.

---

## Integration Example

```python
import os
import cv2
import requests
from datetime import datetime

# Configuration
CAPTURE_FOLDER = "/home/dhruba001/webcam_captures"
BACKEND_URL = "http://192.169.0.111:8000"
DEVICE_ID = "pi_001"

os.makedirs(CAPTURE_FOLDER, exist_ok=True)

def upload_image(image_path):
    """Upload image to backend."""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'source': DEVICE_ID}
            response = requests.post(
                f"{BACKEND_URL}/api/pi/inference/",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['predictions']['class_name']
            confidence = result['predictions']['confidence']
            print(f"✅ {prediction} ({confidence:.2%})")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

# Example: capture and upload
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().isoformat()
        path = os.path.join(CAPTURE_FOLDER, f"frame_{timestamp}.jpg")
        cv2.imwrite(path, frame)
        upload_image(path)
cap.release()
```

---

## Debugging

### Check Health Status

```bash
curl -v http://192.168.1.100:8000/api/pi/health/
```

### Test with cURL Verbose

```bash
curl -v -F "image=@test.jpg" -F "source=test" \
  http://192.168.1.100:8000/api/pi/inference/
```

### Monitor Network Traffic

```bash
# On Pi - capture requests to backend
tcpdump -i eth0 host 192.168.1.100 -v
```

---

## See Also

- [SETUP.md](SETUP.md) - Installation guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- Backend API docs (from your Django project)
