# WasteX - Image Classification System

Django-based waste classification system using Keras/TensorFlow models with operator dashboard.

## 📁 Project Structure

```
wastex/
├── models/                        # ML models directory
│   ├── inception_v3-base-dataset-fe.keras  # Your trained model (not in repo)
│   └── classes.txt                # 9 waste categories
├── media/uploads/                 # Stored images (Miscellaneous Trash only)
├── classifier/                    # Django app
│   ├── models.py                  # Image database model
│   ├── views.py                   # Upload, classify, dashboard views
│   ├── model_loader.py            # Keras model loader
│   └── urls.py                    # URL routing
├── templates/
│   └── classifier/
│       ├── index.html             # Upload interface
│       ├── dashboard.html         # Operator dashboard
│       └── api_docs.html          # API documentation
└── manage.py
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/arf-01/wastex.git
cd wastex
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. **IMPORTANT: Add Your Model File**

⚠️ The model file is **NOT included** in the repository (108 MB - exceeds GitHub limit).

**Download the model and place it in the `models/` directory:**

```
models/
└── inception_v3-base-dataset-fe.keras  (108.33 MB)
```

The model classifies images into 9 waste categories:
- Cardboard
- Food Organics
- Glass
- Metal
- **Miscellaneous Trash** (saved to database)
- Paper
- Plastic
- Textile Trash
- Vegetation

### 4. Configure PostgreSQL Database

Update `wastex/settings.py` with your PostgreSQL credentials:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'your_db_name',
        'USER': 'your_user',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### 5. Run Migrations

```powershell
python manage.py migrate
```

### 6. Run Server

```powershell
python manage.py runserver
```

## 🌐 Available URLs

- **Home**: http://127.0.0.1:8000/
- **Upload Interface**: http://127.0.0.1:8000/classifier/
- **Operator Dashboard**: http://127.0.0.1:8000/classifier/dashboard/
- **API Documentation**: http://127.0.0.1:8000/classifier/api/docs/
- **API Endpoint**: http://127.0.0.1:8000/api/predict/

## 🖼️ Usage

### Web Interface

1. Go to http://127.0.0.1:8000/classifier/
2. Drag and drop an image or click to upload
3. Click "Classify Image"
4. View predictions with confidence scores
5. If classified as "Miscellaneous Trash", image is automatically saved to database

### Operator Dashboard

View all saved Miscellaneous Trash images:
- Navigate to http://127.0.0.1:8000/classifier/dashboard/
- See statistics: total images, average confidence
- Browse paginated image gallery (20 per page)
- View metadata: dimensions, file size, upload time, IP address

### API Endpoint

**POST** `/api/predict/`

```bash
curl -X POST -F "image=@image.jpg" http://localhost:8000/api/predict/
```

Response (if classified as Miscellaneous Trash):
```json
{
  "success": true,
  "predictions": [
    {
      "class": "Miscellaneous Trash",
      "confidence": 87.45,
      "confidence_percent": "87.45%"
    }
  ],
  "top_prediction": {
    "class": "Miscellaneous Trash",
    "confidence": 87.45,
    "confidence_percent": "87.45%"
  },
  "saved_to_database": true,
  "message": "Image classified as Miscellaneous Trash and saved to database"
}
```

## �️ Database Schema

**Table: `images`**

| Field | Type | Description |
|-------|------|-------------|
| id | Integer | Primary key |
| image | CharField | File path (relative to MEDIA_ROOT) |
| filename | CharField | Original filename |
| file_size | Integer | Size in bytes |
| width | Integer | Image width in pixels |
| height | Integer | Image height in pixels |
| top_prediction | CharField | Classification result |
| confidence | Float | Confidence score (0-100) |
| all_predictions | JSON | All class predictions |
| uploaded_at | DateTime | Upload timestamp |
| classified_at | DateTime | Classification timestamp |
| ip_address | GenericIPAddress | Client IP |
| user_agent | TextField | Browser/client info |

## 📝 Key Features

### 🎯 Smart Storage
- **Only "Miscellaneous Trash"** images are saved
- Other waste types (Cardboard, Glass, Metal, Paper, Plastic, etc.) are classified but NOT stored
- Saves storage space and focuses on items needing manual review

### 📂 File Organization
- Images saved to: `media/uploads/YYYY/MM/DD/filename.jpg`
- Database stores relative path only
- Automatic date-based folder structure

### 📊 Operator Dashboard
- Real-time statistics
- Image gallery with thumbnails
- Detailed metadata for each image
- Pagination for large datasets

## ⚙️ How It Works

1. **Upload**: User uploads image via web UI or API
2. **Classification**: Keras model predicts waste category
3. **Conditional Save**:
   - If "Miscellaneous Trash": Save to `media/uploads/` + create database entry
   - Otherwise: Delete temporary file, return prediction only
4. **Dashboard**: Operators review all saved Miscellaneous Trash images

## 🔧 Customization

### Change Input Size

The input size is auto-detected from your model. If you need to change it:

```python
# In classifier/model_loader.py
self.input_shape = (256, 256)  # Your size
```

### Custom Preprocessing

Edit `preprocess_image()` in `classifier/model_loader.py` to match your model's training preprocessing.

## 📦 Production Deployment

Update `settings.py`:

```python
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com']
STATIC_ROOT = BASE_DIR / 'staticfiles'
```

Install gunicorn:

```powershell
pip install gunicorn
gunicorn wastex.wsgi:application
```

## 🐛 Troubleshooting

**Model not loading?**
- Check file is in `models/` directory
- Verify it's `.keras` format
- Check TensorFlow is installed

**Predictions look wrong?**
- Verify `classes.txt` matches your model's training classes
- Check image preprocessing matches training

**Import errors?**
- Activate virtual environment
- Install all requirements: `pip install -r requirements.txt`
