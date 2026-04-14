# Pi Real-Time Processing - Visual Architecture

## System Architecture Diagram

```
╔════════════════════════════════════════════════════════════════════════════╗
║                          COMPLETE DATA FLOW                                ║
╚════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────┐
│                       RASPBERRY PI (192.169.0.111)                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                     WEBCAM CAPTURE                             │    │
│  │                                                                │    │
│  │   Camera → OpenCV/Picamera → Frame Processing → JPEG         │    │
│  └──────────────────────────┬─────────────────────────────────────┘    │
│                             │ saves                                    │
│                             ↓                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │         /home/dhruba001/webcam_captures/                      │    │
│  │                                                                │    │
│  │   capture_20260413_185152_123456.jpg                          │    │
│  │   capture_20260413_185153_123457.jpg                          │    │
│  │   capture_20260413_185154_123458.jpg                          │    │
│  └──────────────────────────┬─────────────────────────────────────┘    │
│                             │ detects                                  │
│                             ↓                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │         pi_image_watcher.py (File Monitor)                    │    │
│  │                                                                │    │
│  │   • Watchdog: inotify on Linux                               │    │
│  │   • Detects: .jpg, .jpeg, .png, .bmp files                   │    │
│  │   • Waits: 500ms for file to complete                        │    │
│  │   • Retries: 3 times with exponential backoff                │    │
│  │   • Logs: Full operation details to /tmp/                    │    │
│  └──────────────────────────┬─────────────────────────────────────┘    │
│                             │ uploads                                  │
└─────────────────────────────┼──────────────────────────────────────────┘
                              │
                ╔═════════════════════════════════╗
                ║  HTTP POST /api/pi/inference/   ║
                ║                                 ║
                ║  Content-Type: multipart/form   ║
                ║  - image: <binary jpeg>         ║
                ║  - source: "pi_001"             ║
                ╚════════════┬══════════════════╝
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              BACKEND SERVER (Your Laptop - 192.168.1.100:8000)            │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Django View: api_pi_inference()                                │   │
│  │                                                                  │   │
│  │  1. Receive multipart/form-data                                │   │
│  │     └─ Extract: image file, source device ID                  │   │
│  │                                                                  │   │
│  │  2. Save to disk                                               │   │
│  │     └─ media/pi_uploads/pi_001/2026-04-13T18:51:52.jpg        │   │
│  │                                                                  │   │
│  │  3. Load ML Model                                              │   │
│  │     └─ Load pre-trained InceptionV3 from models/               │   │
│  │                                                                  │   │
│  │  4. Preprocess Image                                           │   │
│  │     └─ Resize, normalize, prepare input tensor                │   │
│  │                                                                  │   │
│  │  5. Run Inference                                              │   │
│  │     └─ Forward pass through neural network                     │   │
│  │                                                                  │   │
│  │  6. Get Prediction                                             │   │
│  │     └─ class_name: "plastic"                                  │   │
│  │        confidence: 0.9823                                      │   │
│  │                                                                  │   │
│  │  7. Save to Database                                           │   │
│  │     └─ Image model with:                                       │   │
│  │        • source_device: "pi_001"                               │   │
│  │        • predicted_label: "plastic"                            │   │
│  │        • confidence: 0.9823                                    │   │
│  │        • created_at: 2026-04-13T18:51:52Z                     │   │
│  │                                                                  │   │
│  │  8. Return JSON Response (200 OK)                              │   │
│  │     └─ status: "success"                                       │   │
│  │        predictions: {...}                                      │   │
│  │        image_id: 12345                                         │   │
│  │        timestamp: "2026-04-13T18:51:52.123456Z"               │   │
│  └────────────────────────┬─────────────────────────────────────┘   │
│                           │                                          │
└───────────────────────────┼──────────────────────────────────────────┘
                            │
                  ╔═════════════════════════╗
                  ║  HTTP 200 OK            ║
                  ║  JSON Response          ║
                  ║  (instant result)       ║
                  ╚════════════┬════════════╝
                               │
                               ↓
┌──────────────────────────────────────────────────────┐
│         RASPBERRY PI (receives response)             │
│                                                      │
│   pi_image_watcher.py                               │
│   └─ Logs result: "✅ Upload successful!"           │
│      └─ Predictions: {                              │
│           "class_name": "plastic",                  │
│           "confidence": 0.9823                      │
│         }                                           │
│                                                      │
│   Optional: Display result on local screen/log      │
└──────────────────────────────────────────────────────┘
```

## Timeline

```
T=0ms      Image captured by webcam
T=+10ms    Frame saved to JPEG file
T=+12ms    Watcher detects file creation
T=+512ms   File fully written, watcher initiates upload
T=+520ms   HTTP POST sent to backend
T=+600ms   Backend receives request
T=+750ms   Image saved to disk
T=+1200ms  Model loaded into memory
T=+1800ms  Inference complete
T=+1850ms  Result saved to database
T=+1870ms  Response sent to Pi
T=+1950ms  Pi receives result

TOTAL: ~2 seconds from capture to result
```

## Multi-Device Scenario

```
┌─────────────────────────────────────┐
│     Raspberry Pi #1 (pi_001)       │
│     /home/dhruba001/webcam_...    │
│              │                      │
└──────────────┼──────────────────────┘
               │
               │  HTTP POST
               │
┌──────────────┼──────────────────────────────────┐
│              ↓                                   │
│     Backend Server (single instance)            │
│                                                  │
│     Queue/Process all uploads                   │
│     Save to database with source_device="pi_001"│
│     pi_002, pi_003, etc. all work!             │
│                                                  │
└──────────────┬──────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│     Raspberry Pi #2 (pi_002)       │
│     /home/dhruba001/webcam_...    │
│              │                      │
└──────────────┘──────────────────────┘
```

## Database Schema

```
ImageModel
├── id (primary key)
├── image (file path)
│   └─ media/pi_uploads/pi_001/2026-04-13T18:51:52.jpg
├── source_device (char)
│   └─ "pi_001", "pi_002", etc.
├── predicted_label (char)
│   └─ "plastic", "metal", "glass", "paper", "organic"
├── confidence (float)
│   └─ 0.0 to 1.0
├── created_at (datetime)
│   └─ 2026-04-13T18:51:52.123456Z
└── updated_at (datetime)
```

Query examples:
```sql
-- Get latest images from pi_001
SELECT * FROM classifier_image 
WHERE source_device = 'pi_001'
ORDER BY created_at DESC
LIMIT 10;

-- Count classifications from all Pi devices
SELECT source_device, predicted_label, COUNT(*) 
FROM classifier_image
GROUP BY source_device, predicted_label;

-- Get average confidence
SELECT predicted_label, AVG(confidence)
FROM classifier_image
WHERE source_device = 'pi_001'
GROUP BY predicted_label;
```

## Performance Characteristics

```
Single Image Inference:
├─ Network latency:      100-200ms
├─ Server processing:    1200-1800ms
│  ├─ File save:         30-50ms
│  ├─ Model load:        400-600ms (first time only)
│  ├─ Inference:         500-800ms
│  ├─ DB save:           50-100ms
│  └─ Response build:    20-30ms
└─ Total:               ~1.5-2.0 seconds per image

Batch Processing (5 images):
├─ Upload:              500-1000ms
├─ Processing:          ~300ms per image (model loaded once)
├─ DB operations:       ~50ms for all
└─ Total:              ~2.5-3.5 seconds for 5 images
```

## Error Handling

```
Image Upload
    │
    ├─ Network Error?
    │   └─ Retry 3 times with 2s delay
    │   └─ Log: "Connection failed (attempt 1/3)"
    │
    ├─ Server Error (5xx)?
    │   └─ Retry 3 times
    │   └─ Log: "Backend error"
    │
    ├─ Invalid Image?
    │   └─ Log error and continue
    │
    ├─ Model Not Found?
    │   └─ Log error, skip inference
    │   └─ Still save image to database
    │
    └─ Database Error?
        └─ Log error but don't retry
        └─ Image saved to disk
```

## Integration Points

```
Your Webcam Script
    ↓
    save to /home/dhruba001/webcam_captures/
    ↓
pi_image_watcher.py (runs 24/7)
    ↓
    POST to backend
    ↓
Django API endpoint
    ↓
Your Database (PostgreSQL)
    ↓
Django Admin / Dashboard / API
    ↓
Get statistics, view results, retrain models
```

---

**That's the complete system!** Start with PI_QUICK_START.md to get up and running.
