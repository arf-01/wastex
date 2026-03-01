# WasteX — Waste Classification & Retraining System# WasteX - Waste Classification System



Django-based waste classification system using a **Keras InceptionV3** modelDjango-based waste classification system using PyTorch InceptionV3 model with operator dashboard.

with energy-based out-of-distribution (OOD) detection, operator review

dashboard, delta-based dataset versioning, and a full retraining pipeline.## 📁 Project Structure



---```

wastex/

## 📁 Project Structure├── models/                        # ML models directory

│   ├── model.pth                  # PyTorch InceptionV3 model (not in repo)

```│   └── classes.txt                # 9 waste categories

wastex/├── media/uploads/                 # Stored images (Miscellaneous Trash only)

├── manage.py                          # Django management entry point├── classifier/                    # Django app

├── requirements.txt                   # Python dependencies│   ├── models.py                  # Image database model

││   ├── views.py                   # Upload, classify, dashboard views

├── wastex/                            # Django project settings│   ├── model_loader.py            # PyTorch model loader

│   ├── settings.py                    # Configuration (DB, media, datasets)│   └── urls.py                    # URL routing

│   ├── urls.py                        # Root URL routing├── templates/

│   ├── views.py                       # Welcome page│   └── classifier/

│   ├── wsgi.py / asgi.py│       ├── index.html             # Upload interface

│   └── __init__.py│       ├── dashboard.html         # Operator dashboard

││       └── api_docs.html          # API documentation

├── classifier/                        # Main Django app└── manage.py

│   ├── apps.py                        # App config```

│   ├── models.py                      # DB models (Image, TrashCounter, DatasetVersion, …)

│   ├── model_loader.py                # Keras model loading & inference## 🚀 Quick Start

│   ├── urls.py                        # URL routing for all API & page endpoints

│   ├── migrations/                    # Database migrations### 1. Clone Repository

│   ├── templates/classifier/          # HTML templates

│   │   ├── dashboard.html```bash

│   │   ├── upload.htmlgit clone https://github.com/arf-01/wastex.git

│   │   ├── inspect.htmlcd wastex

│   │   └── dataset.html```

│   └── views/                         # View package (split for readability)

│       ├── __init__.py                # Re-exports all view functions### 2. Install Dependencies

│       ├── helpers.py                 # Constants, utilities, shared helpers

│       ├── pages.py                   # HTML page views (dashboard, upload, …)```powershell

│       ├── classification.py          # POST /classify/ — inference endpointpip install -r requirements.txt

│       ├── trash_api.py               # Trash counter APIs (counts, history)```

│       ├── ood_api.py                 # OOD image APIs (list, review, label)

│       └── dataset_api.py            # Dataset versioning APIs**Note**: PyTorch installation will be large (~2.8 GB). If you have NVIDIA GPU, you can use CUDA version for faster inference.

│

├── training/                          # Retraining pipeline### 3. **IMPORTANT: Add Your Model File**

│   ├── __init__.py                    # Package docstring & quick-start

│   ├── config.py                      # TrainingConfig dataclass + paths⚠️ The model file is **NOT included** in the repository (large file).

│   ├── data.py                        # VersionEntry → tf.data.Dataset loaders

│   ├── train.py                       # Model build, compile, fit loop**Place your PyTorch model in the `models/` directory:**

│   ├── evaluate.py                    # Test metrics, confusion matrix, comparison

│   ├── runner.py                      # Orchestrator: data → train → evaluate → save```

│   └── tasks.py                       # Background thread launcher & status helpersmodels/

│└── model.pth  (Your InceptionV3 PyTorch model)

├── models/                            # ML model artefacts```

│   ├── logits_mdl.keras               # Original shipped model (InceptionV3)

│   ├── classes.txt                    # Current class list (one per line)The model classifies images into 9 waste categories:

│   └── versions/                      # Versioned model outputs- Cardboard

│       └── model_v2_20260224_…/       # Example training run artefacts- Food Organics

│           ├── model.keras- Glass

│           ├── best_model.keras- Metal

│           ├── classes.txt- **Miscellaneous Trash** (saved to database)

│           ├── metrics.json- Paper

│           ├── comparison.json- Plastic

│           ├── config.json- Textile Trash

│           ├── training_log.json- Vegetation

│           ├── training_log.csv

│           └── model_summary.txt### 4. Configure PostgreSQL Database

│

├── datasets/                          # Dataset versions (on disk)Update `wastex/settings.py` with your PostgreSQL credentials:

│   └── v1/

│       ├── dataset_train/```python

│       ├── dataset_test/DATABASES = {

│       └── dataset_val/    'default': {

│        'ENGINE': 'django.db.backends.postgresql',

└── media/                             # User uploads (OOD images persist here)        'NAME': 'your_db_name',

    └── uploads/        'USER': 'your_user',

```        'PASSWORD': 'your_password',

        'HOST': 'localhost',

---        'PORT': '5432',

    }

## 🔄 Retraining Pipeline}

```

The full pipeline flows through five stages:

### 5. Run Migrations

```

┌───────────────┐     ┌───────────────┐     ┌──────────────┐```powershell

│ 1. INFERENCE  │────▶│ 2. OOD REVIEW │────▶│ 3. DATASET   │python manage.py migrate

│   & OOD       │     │   (operator)  │     │   VERSIONING │```

│   detection   │     │   label images│     │   (delta)    │

└───────────────┘     └───────────────┘     └──────┬───────┘### 6. Run Server

                                                   │

                      ┌───────────────┐     ┌──────▼───────┐```powershell

                      │ 5. PROMOTE /  │◀────│ 4. RETRAIN   │python manage.py runserver

                      │   SERVE       │     │   & EVALUATE │```

                      └───────────────┘     └──────────────┘

```## 🌐 Available URLs



### Stage 1 — Inference & OOD Detection- **Home**: http://127.0.0.1:8000/

- User uploads image → `POST /classifier/classify/`- **Upload Interface**: http://127.0.0.1:8000/classifier/

- InceptionV3 model outputs raw logits- **Operator Dashboard**: http://127.0.0.1:8000/classifier/dashboard/

- Energy score = −logsumexp(logits) — lower = more confident- **API Documentation**: http://127.0.0.1:8000/classifier/api/docs/

- **In-distribution** (energy ≤ −4.34 AND softmax ≥ 0.7):- **API Endpoint**: http://127.0.0.1:8000/api/predict/

  increment `TrashCounter`, delete file

- **OOD** (energy > threshold OR softmax < 0.7):## 🖼️ Usage

  save `Image` record for operator review

### Web Interface

### Stage 2 — Operator Review

- `GET /classifier/api/ood/` → list unreviewed OOD images1. Go to http://127.0.0.1:8000/classifier/

- `POST /classifier/api/ood/<id>/review/` → mark as reviewed2. Drag and drop an image or click to upload

- `POST /classifier/api/ood/<id>/label/` → assign class label3. Click "Classify Image"

- Labelled images enter the **staging area**4. View predictions with confidence scores

5. If classified as "Miscellaneous Trash", image is automatically saved to database

### Stage 3 — Dataset Versioning (delta-based)

- `POST /classifier/api/dataset/register-version/` → register existing### Operator Dashboard

  on-disk folder (e.g. `datasets/v1`)

- `GET /classifier/api/dataset/staged/` → view labelled-but-unadded imagesView all saved Miscellaneous Trash images:

- `POST /classifier/api/dataset/create-version/` → create new version:- Navigate to http://127.0.0.1:8000/classifier/dashboard/

  - Inherits parent's `VersionEntry` rows (DB only, no file copy)- See statistics: total images, average confidence

  - Adds staged OOD images as new entries- Browse paginated image gallery (20 per page)

  - Refreshes cached stats, auto-activates- View metadata: dimensions, file size, upload time, IP address



### Stage 4 — Retraining### API Endpoint

```python

from training.config import TrainingConfig**POST** `/api/predict/`

from training.runner import run_training

```bash

config = TrainingConfig(dataset_version="v2", epochs=20)curl -X POST -F "image=@image.jpg" http://localhost:8000/api/predict/

run = run_training(config)   # synchronous, returns TrainingRun record```

```

Response (if classified as Miscellaneous Trash):

Or in a background thread:```json

```python{

from training.tasks import start_training  "success": true,

run = start_training(config)  # non-blocking  "predictions": [

```    {

      "class": "Miscellaneous Trash",

The pipeline:      "confidence": 87.45,

1. Loads train/val/test splits from `VersionEntry` → `tf.data.Dataset`      "confidence_percent": "87.45%"

2. Builds model (fine-tune existing or fresh InceptionV3 backbone)    }

3. Phase 1: frozen base layers → train head only  ],

4. Phase 2: unfreeze all → full fine-tuning at lower LR  "top_prediction": {

5. Early stopping on `val_loss`    "class": "Miscellaneous Trash",

6. Evaluates on test split → accuracy, per-class F1, confusion matrix    "confidence": 87.45,

7. Compares against previous model → promote / keep recommendation    "confidence_percent": "87.45%"

  },

### Stage 5 — Promotion  "saved_to_database": true,

- If `auto_promote=True` and the new model outperforms, it becomes active  "message": "Image classified as Miscellaneous Trash and saved to database"

- All artefacts saved under `models/versions/<run_name>/`}

```

---

## �️ Database Schema

## 🚀 Quick Start

**Table: `images`**

### 1. Clone & install

| Field | Type | Description |

```bash|-------|------|-------------|

git clone https://github.com/arf-01/wastex.git| id | Integer | Primary key |

cd wastex| image | CharField | File path (relative to MEDIA_ROOT) |

python -m venv venv| filename | CharField | Original filename |

venv\Scripts\activate          # Windows| file_size | Integer | Size in bytes |

pip install -r requirements.txt| width | Integer | Image width in pixels |

```| height | Integer | Image height in pixels |

| top_prediction | CharField | Classification result |

### 2. Add your model| confidence | Float | Confidence score (0-100) |

| all_predictions | JSON | All class predictions |

Place your trained `.keras` model in the `models/` directory:| uploaded_at | DateTime | Upload timestamp |

```| classified_at | DateTime | Classification timestamp |

models/| ip_address | GenericIPAddress | Client IP |

├── logits_mdl.keras    # InceptionV3 logits model| user_agent | TextField | Browser/client info |

└── classes.txt         # One class name per line

```## 📝 Key Features



### 3. Configure database### 🎯 Smart Storage

- **Only "Miscellaneous Trash"** images are saved

Update `wastex/settings.py` with your PostgreSQL credentials:- Other waste types (Cardboard, Glass, Metal, Paper, Plastic, etc.) are classified but NOT stored

```python- Saves storage space and focuses on items needing manual review

DATABASES = {

    'default': {### 📂 File Organization

        'ENGINE': 'django.db.backends.postgresql',- Images saved to: `media/uploads/YYYY/MM/DD/filename.jpg`

        'NAME': 'wastex',- Database stores relative path only

        'USER': 'postgres',- Automatic date-based folder structure

        'PASSWORD': 'your_password',

        'HOST': 'localhost',### 📊 Operator Dashboard

        'PORT': '5432',- Real-time statistics

    }- Image gallery with thumbnails

}- Detailed metadata for each image

```- Pagination for large datasets



### 4. Migrate & run## ⚙️ How It Works



```bash1. **Upload**: User uploads image via web UI or API

python manage.py migrate2. **Classification**: Keras model predicts waste category

python manage.py runserver3. **Conditional Save**:

```   - If "Miscellaneous Trash": Save to `media/uploads/` + create database entry

   - Otherwise: Delete temporary file, return prediction only

---4. **Dashboard**: Operators review all saved Miscellaneous Trash images



## 🌐 Available Pages## 🔧 Customization



| URL | Description |### Change Input Size

|-----|-------------|

| `/` | Redirect → dashboard |The input size is auto-detected from your model. If you need to change it:

| `/classifier/dashboard/` | Main operator dashboard |

| `/classifier/upload/` | Image upload & classification |```python

| `/classifier/inspect/` | OOD image review & labelling |# In classifier/model_loader.py

| `/classifier/dataset/` | Dataset version browser |self.input_shape = (256, 256)  # Your size

| `/admin/` | Django admin |```



---### Custom Preprocessing



## 📡 API EndpointsEdit `preprocess_image()` in `classifier/model_loader.py` to match your model's training preprocessing.



| Method | Endpoint | Description |## 📦 Production Deployment

|--------|----------|-------------|

| `POST` | `/classifier/classify/` | Upload & classify an image |Update `settings.py`:

| `GET` | `/classifier/api/counts/` | Current trash counts per class |

| `GET` | `/classifier/api/history/` | Trash count time-series |```python

| `GET` | `/classifier/api/ood/` | List OOD images (paginated) |DEBUG = False

| `POST` | `/classifier/api/ood/<id>/review/` | Mark image as reviewed |ALLOWED_HOSTS = ['your-domain.com']

| `POST` | `/classifier/api/ood/<id>/label/` | Assign label to image |STATIC_ROOT = BASE_DIR / 'staticfiles'

| `GET` | `/classifier/api/classes/` | List all known classes |```

| `GET` | `/classifier/api/dataset/versions/` | List dataset versions |

| `GET` | `/classifier/api/dataset/active/` | Get active version |Install gunicorn:

| `POST` | `/classifier/api/dataset/set-active/` | Set active version |

| `GET` | `/classifier/api/dataset/staged/` | List staged images |```powershell

| `POST` | `/classifier/api/dataset/create-version/` | Create new version (delta) |pip install gunicorn

| `POST` | `/classifier/api/dataset/register-version/` | Register on-disk folder |gunicorn wastex.wsgi:application

| `GET` | `/classifier/api/dataset/images/` | Browse images in a version |```



---## 🐛 Troubleshooting



## 🗂️ Database Models**Model not loading?**

- Check file is in `models/` directory

| Model | Purpose |- Verify it's `.keras` format

|-------|---------|- Check TensorFlow is installed

| `DatasetVersion` | Versioned dataset snapshots with cached stats |

| `VersionEntry` | Delta-based image membership (no disk duplication) |**Predictions look wrong?**

| `DatasetClass` | Canonical growing registry of waste class labels |- Verify `classes.txt` matches your model's training classes

| `Image` | Uploaded images with OOD metadata and review fields |- Check image preprocessing matches training

| `TrashCounter` | Per-class item counts as time-series |

| `TrainingRun` | Full training lifecycle: config → train → evaluate → promote |**Import errors?**

- Activate virtual environment

---- Install all requirements: `pip install -r requirements.txt`


## ⚙️ Waste Categories

Default classes (from `models/classes.txt`):
- Cardboard · Food Organics · Glass · Metal
- Paper · Plastic · Textile Trash · Vegetation

New classes are added automatically when operators label OOD images.
