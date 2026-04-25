# ♻️ WasteX

**WasteX** is an edge-to-cloud federated waste classification system. It empowers edge devices (like Raspberry Pis) to perform real-time trash classification while syncing out-of-distribution (OOD) data to a central cloud dashboard for operator review, dataset versioning, and automated model retraining.

---

## ✨ Key Features

- **🧠 Edge Inference**: Fast, local waste classification using a TensorFlow/Keras InceptionV3 model.
- **☁️ Cloud Dashboard**: Centralized management portal for reviewing edge telemetry and unclassified items.
- **🛡️ Out-of-Distribution (OOD) Detection**: Automatically flags uncertain classifications using Energy-based scoring for human review.
- **🔄 Automated Retraining Pipeline**: Operators can label OOD images, version the dataset, and kick off a background retraining pipeline to improve the model.
- **🔐 Role-Based Access**: Separate dashboards for `Edge` nodes (data collection) and `Master` operators (model retraining & dataset management).

---

## 🏗️ Architecture

WasteX is built to run in two environments using the exact same codebase, determined by the `SITE_ROLE` environment variable:

1. **CLOUD Mode** (`SITE_ROLE=CLOUD`):
   - Runs on cloud platforms (e.g., Render) connected to PostgreSQL and S3-compatible storage (Backblaze B2).
   - Serves as the central data broker, dashboard, and retraining hub.
   - *Skips loading heavy ML models into memory on startup to save cloud resources.*
2. **EDGE Mode** (`SITE_ROLE=EDGE`):
   - Runs locally on devices like Raspberry Pis.
   - Loads the TensorFlow model into memory for fast, real-time inference using a webcam.
   - Syncs data back to the Cloud broker using secure API keys.

---

## 🚀 Quick Start (Local / Edge)

### 1. Clone & Install
```bash
git clone https://github.com/arf-01/wastex.git
cd wastex
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Add the Initial Model
Ensure your trained Keras model is placed in the `models/` directory:
```text
models/
├── logits_mdl.keras    # Your Keras InceptionV3 model
└── classes.txt         # List of waste classes
```

### 3. Run the Local Server
```bash
python manage.py migrate
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` to view the local dashboard!

---

## ☁️ Cloud Deployment (Render)

WasteX is optimized for deployment on **Render** using a PostgreSQL database and Backblaze B2 for media storage.

1. Create a **Web Service** on Render connected to your GitHub repo.
2. Set the **Start Command** to: `gunicorn wastex.wsgi:application --bind 0.0.0.0:$PORT`
3. Add the following **Environment Variables**:
   - `SITE_ROLE`: `CLOUD`
   - `DEBUG`: `False`
   - `DATABASE_URL`: Your Supabase/PostgreSQL connection string
   - `DB_SSL_MODE`: `require`
   - `SECRET_KEY`: A secure random string
   - *(Plus your B2 Storage credentials)*
4. **Deploy!** The repository's `render-build.sh` script will automatically install dependencies, collect static files, and run database migrations.

---

## 🔐 Roles & Authentication

WasteX relies on Django's Group-based Role Access Control (RBAC):
- **Master User**: Has access to the Retraining Pipeline to manage datasets and trigger model training. Must belong to the `MasterUsers` group.
- **Edge User**: Used by Raspberry Pis for API access and viewing local edge stats. Must belong to the `EdgeUsers` group.

---

*Built for a cleaner, smarter, and more sustainable future.* 🌍
