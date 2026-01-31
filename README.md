# Image Classifier - Keras Model Deployment

Simple Django app for deploying .keras format image classification models.

## 📁 Project Structure

```
wastex/
├── models/              # Place your .keras model here
│   ├── model.keras     # Your trained model
│   └── classes.txt     # Class names (one per line)
├── media/              # Temporary uploads
├── classifier/         # App code
├── templates/
│   └── classifier/
│       └── index.html
└── manage.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Add Your Model

Place your trained `.keras` model in the `models/` directory:

```
models/
└── model.keras
```

### 3. Add Class Names

Create `models/classes.txt` with one class name per line:

```
cat
dog
bird
```

### 4. Run Server

```powershell
python manage.py runserver
```

Visit http://127.0.0.1:8000/classifier/

## 🖼️ Usage

### Web Interface

1. Go to http://127.0.0.1:8000/classifier/
2. Drag and drop an image or click to upload
3. Click "Classify Image"
4. View predictions with confidence scores

### API Endpoint

**POST** `/api/predict/`

```bash
curl -X POST -F "image=@image.jpg" http://localhost:8000/api/predict/
```

Response:
```json
{
  "success": true,
  "predictions": [
    {
      "class": "cat",
      "confidence": 0.95,
      "confidence_percent": "95.00%"
    }
  ],
  "top_prediction": {
    "class": "cat",
    "confidence": 0.95,
    "confidence_percent": "95.00%"
  }
}
```

## 📝 Model Requirements

- Format: `.keras`
- Input: RGB images (any size, will be resized)
- Output: Softmax probabilities
- Default input size: 224x224 (auto-detected from model)

## ⚙️ How It Works

1. **Model Loader** (`classifier/model_loader.py`):
   - Automatically finds `.keras` files in `models/` directory
   - Loads class names from `classes.txt`
   - Preprocesses images (resize, normalize)
   - Returns top-K predictions

2. **Views** (`classifier/views.py`):
   - `index()`: Upload interface
   - `classify()`: Process uploads and return JSON
   - `api_predict()`: CSRF-exempt API endpoint

3. **Demo Mode**:
   - If no model found, returns demo predictions
   - Allows testing without a trained model

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
