# File Upload System: Complete Guide
## How files flow, where they're stored, validation, and control strategies

---

## 1. UPLOAD FLOW: Step-by-Step

### 1.1 User Uploads Image (Upload Page)

**Frontend (browser):**
```
User clicks "Choose File" or drags image onto dropzone
  ↓
JavaScript: handleFile(file)
  ├─ Validate file type: must be image/* (JPG, PNG, GIF, WEBP)
  ├─ Check file size: must be < 50MB (browser-side check)
  ├─ Show preview: FileReader reads file as Data URL
  └─ User clicks "Classify" → classifyImage()
       ↓
       Create FormData object: { image: file }
       ↓
       POST /classifier/classify/
       ├─ Header: Content-Type: multipart/form-data (automatic)
       ├─ Body: Binary file data (actual image bytes)
       └─ Request size: ~file size + headers (~1MB for 1MB image)
```

**Backend (Django):**
```
Receive POST /classifier/classify/
  ↓
Django parses multipart/form-data
  ├─ Extract "image" field
  ├─ Get UploadedFile object (wrapper around binary stream)
  └─ File is in memory (not yet on disk)
       ↓
       Backend: classify() view
       ├─ Validate file again (server-side check)
       │  ├─ Is it actually an image? (check magic bytes: JPEG=FFD8FF, PNG=89504E47)
       │  ├─ File size < 50MB?
       │  ├─ File size > 0 bytes?
       │  └─ If any fail → return error JSON { error: "Invalid file" }
       │
       ├─ Save file to disk using Django FileField
       │  └─ This triggers: media_root/uploads/YYYY/MM/DD/filename.jpg
       │     (Django auto-creates date-based subdirectories)
       │
       ├─ Run inference on saved image
       │  ├─ Load image from disk (Pillow library)
       │  ├─ Resize to 299×299
       │  ├─ Normalize pixel values
       │  └─ Model inference (TensorFlow) → logits
       │
       ├─ Detect OOD
       │  └─ Compute energy score, check threshold
       │
       ├─ Save metadata to database
       │  └─ INSERT INTO classifier_image (path, label, energy, ood, timestamp)
       │     └─ path = "uploads/2026/03/29/abc123.jpg" (relative to MEDIA_ROOT)
       │
       └─ Return JSON response
            ↓
            { 
              predicted_class: "Plastic",
              confidence: 0.94,
              logits: [2.34, 1.23, -0.45, ...],
              energy: 2.345,
              ood: false,
              saved_to_db: true,
              file_path: "uploads/2026/03/29/abc123.jpg"
            }
```

**Frontend (receives response):**
```
showResult(data)
  ├─ Display predicted class
  ├─ Show logits as bars
  ├─ Show energy score
  ├─ Show OOD flag if detected
  └─ showToast("Classified as Plastic")
```

---

## 2. WHERE FILES ARE STORED

### 2.1 Directory Structure
```
Project Root: C:\WASTE\wastex\
  ↓
media/                                    ← MEDIA_ROOT (configurable via .env)
  └─ uploads/                             ← All user uploads go here
     └─ YYYY/MM/DD/                       ← Date-based subdirectories
        ├─ 2026/
        │  ├─ 03/
        │  │  ├─ 29/
        │  │  │  ├─ image_abc123.jpg     ← Uploaded image
        │  │  │  ├─ photo_def456.png
        │  │  │  └─ waste_xyz789.jpg
        │  │  ├─ 28/
        │  │  └─ 27/
        │  ├─ 02/
        │  └─ 01/
```

### 2.2 Default vs. Custom Paths

**Default (hardcoded):**
```python
# Before .env support
MEDIA_ROOT = BASE_DIR / 'media'  # C:\WASTE\wastex\media
```
❌ Problem: All images stored on C: drive → fills up C: drive.

**With .env Support (Current):**
```python
# After .env support (settings.py)
MEDIA_ROOT = Path(os.getenv('WASTE_MEDIA_ROOT', BASE_DIR / 'media'))
```

**In .env file:**
```env
# Default (if not set): C:\WASTE\wastex\media
# To move to another drive:
WASTE_MEDIA_ROOT=D:/Projects/WasteData/media
```

**Result:** User edits `.env` one time, all images go to D: drive automatically.

### 2.3 Why Date-Based Subdirectories?
```
Why organize uploads/2026/03/29/?

1. Performance:
   ├─ 1 image: 1 directory has 1 file (fast to list)
   ├─ 100k images all in 1 folder: filesystem is slow (thousands of files/dir)
   └─ 100k images in subdirs: each has ~10 files (fast)

2. Manageability:
   ├─ Easy to find old images: "images from March 29"
   ├─ Easy to delete old data: "delete 2026/01/" (old data)
   └─ Backup efficiency: can exclude recent uploads folder

3. Scalability:
   ├─ 1000 images/day = manageable
   ├─ 1M images/day = manageable (with date partitioning)
   └─ 1M images in 1 folder = filesystem hangs
```

---

## 3. FILE NAMING & COLLISION PREVENTION

### 3.1 Current Approach
**Django FileField Auto-Generates Names:**
```python
# In models.py
class Image(models.Model):
    image_file = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    # Django auto-generates: uploads/2026/03/29/filename_abc123.jpg

# Django naming:
# If user uploads "photo.jpg" twice on same day:
#   1st: uploads/2026/03/29/photo.jpg
#   2nd: uploads/2026/03/29/photo_abc123.jpg  ← Django adds suffix
```

**Why Suffix?**
```
If two users upload "photo.jpg" simultaneously:
  ├─ File 1: photo.jpg
  ├─ File 2: photo_xyz789.jpg  ← Conflict prevented by Django
  └─ Both saved, no data loss
```

### 3.2 Better: Hash-Based Naming
**What Could Be Better:**
```python
# Current: Django auto-name (collision-safe but non-deterministic)
uploads/2026/03/29/photo_abc123.jpg

# Better: Hash-based (deterministic, collision-free)
def get_upload_path(instance, filename):
    import hashlib
    file_hash = hashlib.md5(instance.image_file.read()).hexdigest()[:8]
    return f'uploads/{file_hash[:2]}/{file_hash[2:]}/{filename}'
    # Result: uploads/a1/b2c3d4e5/photo.jpg
    # Benefits:
    #   - Same file → same hash → if re-uploaded, reused
    #   - Shards files (a1/, a2/, ... reduces files per dir)
    #   - No duplicate storage if identical image uploaded twice
```

**Trade-off:**
- ✅ Pros: Deduplication, faster lookup.
- ❌ Cons: Requires hashing every upload (slower by ~50ms).

---

## 4. FILE VALIDATION & SECURITY

### 4.1 What Gets Validated?

**Frontend Validation (JavaScript):**
```javascript
function handleFile(file) {
    // 1. File type check
    if (!file.type.startsWith('image/')) {
        showToast('Please select an image file', 'error');
        return;
    }
    
    // 2. File size check
    const MAX_SIZE = 50 * 1024 * 1024;  // 50MB
    if (file.size > MAX_SIZE) {
        showToast(`File too large (max ${MAX_SIZE / 1024 / 1024}MB)`, 'error');
        return;
    }
    
    // 3. Continue if valid
    // ...
}
```

**Why Browser-Side?**
- ✅ Instant feedback (no server round-trip).
- ✅ Saves bandwidth (don't upload if invalid).
- ✅ Better UX (no wait).

**BUT: Browser Validation is NOT Secure** ❌
- User can disable JavaScript.
- User can forge requests with curl/Postman.
- Browser validation is "user convenience only."

---

### 4.2 Backend Validation (Server-Side, Required)

**In Django View:**
```python
from PIL import Image as PILImage
from django.core.exceptions import ValidationError

def classify(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image provided'}, status=400)
    
    uploaded_file = request.FILES['image']
    
    # ── Validation 1: File size ──
    MAX_SIZE = 50 * 1024 * 1024  # 50MB
    if uploaded_file.size > MAX_SIZE:
        return JsonResponse({
            'error': f'File too large (max {MAX_SIZE / 1024 / 1024}MB)'
        }, status=400)
    
    if uploaded_file.size == 0:
        return JsonResponse({'error': 'Empty file'}, status=400)
    
    # ── Validation 2: Is it actually an image? ──
    try:
        img = PILImage.open(uploaded_file)
        img.verify()  # Validate image format
        # Check dimensions (not required, but can be done)
        if img.size[0] < 50 or img.size[1] < 50:
            raise ValidationError('Image too small (min 50×50)')
    except Exception as e:
        return JsonResponse({'error': f'Invalid image: {str(e)}'}, status=400)
    
    # ── Validation 3: File extension ──
    allowed_exts = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext not in allowed_exts:
        return JsonResponse({'error': f'File type not allowed: {file_ext}'}, status=400)
    
    # ── Validation 4: Check for malicious uploads ──
    # (E.g., user renames .exe to .jpg)
    if not verify_image_magic_bytes(uploaded_file):
        return JsonResponse({'error': 'File is not a valid image'}, status=400)
    
    # ── All checks passed, safe to process ──
    # ... continue with inference ...
```

**Magic Byte Verification:**
```python
def verify_image_magic_bytes(file_obj):
    """Check file header to confirm it's actually an image."""
    file_obj.seek(0)
    header = file_obj.read(12)
    
    # JPEG: FF D8 FF
    if header[:3] == b'\xff\xd8\xff':
        return True
    
    # PNG: 89 50 4E 47
    if header[:4] == b'\x89PNG':
        return True
    
    # GIF: 47 49 46
    if header[:3] == b'GIF':
        return True
    
    # WEBP: RIFF ... WEBP
    if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return True
    
    return False
```

**Why This Matters:**
```
Attack vector: User uploads malicious.exe, renames to photo.jpg
  ├─ Frontend check passes (it ends in .jpg)
  ├─ Backend extension check passes
  ├─ BUT: Magic byte check fails (it's really an .exe)
  └─ Server rejects safely

Without magic byte check:
  ├─ File gets saved to disk as .jpg
  ├─ Server executes it (if running Windows, might auto-execute)
  └─ Security breach!
```

---

## 5. STORAGE LIMITS & QUOTAS

### 5.1 Why Control Storage?

**Scenario:**
```
Operator uploads images continuously for 1 month
  ├─ 1000 images/day
  ├─ ~2MB per image (average)
  ├─ Total: 1000 × 2MB × 30 days = 60GB
  ├─ D: drive has 100GB → still OK
  ├─ But C: drive (default) → FULL after 10 days
  └─ App crashes: "Disk full"
```

**Why Control:**
- ✅ Prevent disk exhaustion.
- ✅ Give operator warning (90% full).
- ✅ Auto-clean old data (retention policy).
- ✅ Monitor usage trends.

### 5.2 Storage Policies (What Could Be Better)

**Current (No Limits):**
```python
# No checks → unlimited uploads
if uploaded_file.size > 50MB:  # File-level check only
    reject()
# No quota check
```

**Better: Implement Quota System**
```python
def check_storage_quota(new_file_size):
    """Check if adding this file exceeds quota."""
    
    QUOTA_GB = 500  # Max 500GB of uploads
    WARN_THRESHOLD = 0.9  # Warn at 90%
    
    # 1. Get current usage
    import shutil
    total, used, free = shutil.disk_usage(MEDIA_ROOT)
    current_size = used  # Rough estimate (should be precise)
    
    # 2. Check if quota exceeded
    if current_size + new_file_size > QUOTA_GB * 1e9:
        return False, f"Storage quota exceeded ({QUOTA_GB}GB)"
    
    # 3. Check if approaching limit
    usage_pct = current_size / (QUOTA_GB * 1e9)
    if usage_pct > WARN_THRESHOLD:
        log_warning(f"Storage {usage_pct*100:.0f}% full")
    
    return True, None

# In classify view:
ok, error = check_storage_quota(uploaded_file.size)
if not ok:
    return JsonResponse({'error': error}, status=507)  # 507 = Storage Exhausted
```

**Better: Retention Policy**
```python
def cleanup_old_uploads(days_to_keep=90):
    """Delete uploads older than N days."""
    from django.utils import timezone
    from datetime import timedelta
    
    cutoff_date = timezone.now() - timedelta(days=days_to_keep)
    old_images = Image.objects.filter(created_at__lt=cutoff_date)
    
    for image in old_images:
        # Delete from disk
        if os.path.exists(image.image_file.path):
            os.remove(image.image_file.path)
        # Delete from DB
        image.delete()
    
    return len(old_images)

# Run daily (via Celery or cron):
# cleanup_old_uploads(days_to_keep=90)
```

---

## 6. IMAGE COMPRESSION & OPTIMIZATION

### 6.1 Current Approach (No Optimization)
```python
# User uploads 10MB PNG → saved as 10MB PNG
# Storage: 10MB
# Bandwidth: 10MB download
# Problem: Wasteful
```

### 6.2 Better: Compress on Upload
```python
from PIL import Image as PILImage
from io import BytesIO

def compress_image(uploaded_file, quality=80):
    """Compress image and save as JPEG."""
    
    # 1. Open original
    img = PILImage.open(uploaded_file)
    
    # 2. Convert RGBA/WEBP to RGB (JPEG compatibility)
    if img.mode in ('RGBA', 'LA', 'P'):
        background = PILImage.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    
    # 3. Compress to JPEG
    output = BytesIO()
    img.save(output, format='JPEG', quality=quality, optimize=True)
    output.seek(0)
    
    return output

# In classify view:
uploaded_file = request.FILES['image']
compressed = compress_image(uploaded_file, quality=80)

# Save compressed version
instance.image_file.save(
    f'{filename}.jpg',
    compressed,
    save=False
)
```

**Storage Impact:**
```
Before: 10MB PNG → 10MB stored
After:  10MB PNG → 1-2MB JPEG (80% compression!)

Total storage for 100k images:
  Before: 100k × 10MB = 1TB
  After:  100k × 1.5MB = 150GB (6× saving!)
```

**Trade-off:**
- ✅ Pros: 60-80% storage saving, faster download.
- ❌ Cons: JPEG is lossy (imperceptible quality loss for photos).

---

## 7. UPLOAD WORKFLOW: Dashboard to Database

### 7.1 Complete Data Flow
```
┌─ User (Browser) ──────────────────────────┐
│ 1. Select image file (e.g., waste.jpg)    │
│ 2. Click "Classify"                       │
└─ (Binary file data)──────────────────────┘
                ↓ 
     POST /classifier/classify/
     (multipart/form-data)
                ↓
┌─ Django View (classify.py) ───────────────┐
│ 1. Receive UploadedFile                   │
│ 2. Validate:                              │
│    ├─ File type (image/*)                 │
│    ├─ File size (< 50MB)                  │
│    ├─ Magic bytes (actually an image?)    │
│    └─ Infection scan (if using ClamAV)    │
│ 3. Optionally compress (lossy JPEG)       │
│ 4. Save to disk:                          │
│    └─ media/uploads/2026/03/29/waste.jpg  │
│ 5. Create Image model instance            │
│ 6. Run inference                          │
│ 7. Save to DB                             │
└─ (Metadata + path)───────────────────────┘
                ↓
┌─ Database (SQLite/Postgres) ──────────────┐
│ INSERT INTO classifier_image:             │
│ {                                         │
│   id: 12345,                              │
│   image_file: "uploads/2026/03/29/waste", │
│   predicted_label: "Plastic",             │
│   energy: 2.345,                          │
│   ood: False,                             │
│   timestamp: "2026-03-29 14:32:15",       │
│   ...                                     │
│ }                                         │
└───────────────────────────────────────────┘
                ↓
┌─ Response (JSON) ──────────────────────────┐
│ {                                          │
│   "predicted_class": "Plastic",            │
│   "confidence": 0.94,                      │
│   "logits": [2.34, 1.23, ...],             │
│   "energy": 2.345,                        │
│   "ood": false,                           │
│   "saved_to_db": true                     │
│ }                                          │
└─ Frontend displays result ──────────────────┘
```

---

## 8. UPLOAD RATE LIMITING (Prevent Abuse)

### 8.1 Why Rate Limit?

**Attack Scenario:**
```
Malicious bot uploads continuously:
  ├─ 1000 requests/second
  ├─ Each runs inference (CPU 100%)
  ├─ Server becomes unresponsive
  ├─ Legitimate users can't classify
  └─ Disk fills up with 1000 images/second
```

**With Rate Limiting:**
```
Same bot, but with rate limit (10 requests/minute/IP):
  ├─ 1st request: ✓ allowed
  ├─ 2nd-11th requests: ✓ allowed
  ├─ 11th request: ✗ rejected (429 Too Many Requests)
  ├─ Must wait 1 minute
  └─ Legitimate users not impacted
```

### 8.2 Implementation (Django Ratelimit)

```python
from django_ratelimit.decorators import ratelimit

@ratelimit(key='ip', rate='10/m', method='POST')
def classify(request):
    """10 requests per minute per IP."""
    # ... rest of code ...
```

**Or Manual:**
```python
from django.views.decorators.cache import cache_page
from django.core.cache import cache

def classify(request):
    ip = request.META.get('REMOTE_ADDR')
    cache_key = f'classify_rate_{ip}'
    
    # Get request count for this IP in last 60s
    count = cache.get(cache_key, 0)
    
    if count >= 10:  # Max 10 requests/minute
        return JsonResponse(
            {'error': 'Rate limit exceeded'},
            status=429
        )
    
    # Increment counter
    cache.set(cache_key, count + 1, timeout=60)
    
    # ... rest of code ...
```

---

## 9. BEST PRACTICES CHECKLIST

| Item | Current | Status | Why |
|------|---------|--------|-----|
| **Frontend validation** | ✅ | Done | Fast UX feedback |
| **Backend validation** | ✅ | Done | Security (server-side required) |
| **Magic byte check** | ❌ | TODO | Prevent malicious file uploads |
| **Size limit** | ✅ (50MB) | Done | Prevent DoS |
| **Date-based dirs** | ✅ | Done | Filesystem performance |
| **Disk quota check** | ❌ | TODO | Prevent disk exhaustion |
| **Compression** | ❌ | TODO | Save 60-80% storage |
| **Retention policy** | ❌ | TODO | Auto-clean old files |
| **Rate limiting** | ❌ | TODO | Prevent abuse |
| **HTTPS only** | ❌ | TODO | Encrypt uploads in transit |
| **Virus scan** | ❌ | TODO | ClamAV integration (advanced) |
| **Access logs** | ❌ | TODO | Audit trail |

---

## 10. SUMMARY

### Where Files Are Stored:
- **Default:** `C:\WASTE\wastex\media\uploads\2026\03\29\image.jpg`
- **Configurable:** Edit `.env` → `WASTE_MEDIA_ROOT=D:/Data`

### Why This Matters:
- ✅ Date-based partitioning (filesystem performance).
- ✅ Configurable root (multi-drive support, installer flexibility).
- ✅ FileField auto-handling (Django manages naming, collisions).

### What Gets Validated:
1. **Frontend:** File type, size (UX).
2. **Backend:** File type, size, magic bytes, image validity (security).

### What Could Be Better:
1. **High Priority:** Disk quota checks, compression, magic byte verification.
2. **Medium Priority:** Rate limiting, retention policy, access logs.
3. **Low Priority:** Virus scanning, distributed storage, CDN delivery.

### Next Action:
If you want to add one feature today:
- **Best ROI:** Add magic byte verification (30 mins, high security benefit).
- **Best UX:** Add compression (1 hr, saves 60% disk).
- **Best Ops:** Add disk quota check (1 hr, prevents crashes).

---

**End of File Upload Guide**
