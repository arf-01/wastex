#!/usr/bin/env python3
"""
Raspberry Pi Real-time Image Upload Service

This script monitors a folder on the Raspberry Pi for new images
and automatically uploads them to the backend for inference.

Setup:
1. Place this script on your Raspberry Pi
2. Install watchdog: pip install watchdog requests
3. Run: python3 image_watcher.py

Configuration:
- Modify WATCH_FOLDER to point to your webcam capture folder
- Modify BACKEND_URL to match your backend server
- Modify SOURCE_ID to identify this Pi device
"""

import os
import sys
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Folder to watch for new images (where your webcam script saves images)
WATCH_FOLDER = "/home/dhruba001/webcam_captures"

# Backend server URL (change to your actual backend IP/domain)
BACKEND_URL = "http://192.169.0.111:8000"  # Windows backend on WiFi

# Unique identifier for this Raspberry Pi device
SOURCE_ID = "pi_001"

# Image file extensions to watch for
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Logging configuration
LOG_FILE = f"/tmp/pi_image_watcher_{SOURCE_ID}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE WATCHER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ImageUploadHandler(FileSystemEventHandler):
    """Watch folder and upload new images to backend."""
    
    def on_created(self, event):
        """Called when a new file is created in the watched folder."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        file_ext = Path(file_path).suffix.lower()
        
        # Only process image files
        if file_ext not in VALID_EXTENSIONS:
            return
        
        # Small delay to ensure file is fully written
        time.sleep(0.5)
        
        logger.info(f"📷 New image detected: {file_path}")
        self.upload_image(file_path)
    
    def upload_image(self, file_path):
        """Upload image to backend for inference."""
        
        # Check if file still exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return
        
        # Retry logic
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with open(file_path, 'rb') as image_file:
                    files = {'image': image_file}
                    data = {'source': SOURCE_ID}
                    
                    # Send to backend
                    response = requests.post(
                        f"{BACKEND_URL}/classifier/api/pi/inference/",
                        files=files,
                        data=data,
                        timeout=30
                    )
                    
                    # Check response
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"✅ Upload successful!")
                        logger.info(f"   Predicted Class: {result.get('predicted_class')}")
                        logger.info(f"   Saved to DB: {result.get('saved_to_db')}")
                        return  # Success!
                    else:
                        logger.error(f"❌ Backend error (attempt {attempt}/{MAX_RETRIES})")
                        logger.error(f"   Status: {response.status_code}")
                        logger.error(f"   Response: {response.text}")
                
            except requests.exceptions.ConnectionError:
                logger.error(f"⚠️  Connection failed (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
            
            except Exception as e:
                logger.error(f"❌ Unexpected error (attempt {attempt}/{MAX_RETRIES})")
                logger.error(f"   {type(e).__name__}: {str(e)}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        
        logger.error(f"💥 Failed to upload after {MAX_RETRIES} attempts: {file_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEALTH CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_backend_health():
    """Verify backend is online before starting."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/classifier/api/pi/health/",
            timeout=5
        )
        if response.status_code == 200:
            logger.info("✅ Backend is online and ready!")
            return True
    except Exception as e:
        logger.warning(f"⚠️  Backend health check failed: {e}")
    
    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Start the file watcher service."""
    
    logger.info("=" * 70)
    logger.info("🚀 Raspberry Pi Image Watcher Service Started")
    logger.info("=" * 70)
    
    # Validate configuration
    if BACKEND_URL == "http://192.168.x.x:8000":
        logger.error("❌ ERROR: Please update BACKEND_URL in this script!")
        logger.error("   Edit the BACKEND_URL variable with your backend's IP address")
        sys.exit(1)
    
    # Create watch folder if it doesn't exist
    watch_path = Path(WATCH_FOLDER)
    watch_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Watching folder: {WATCH_FOLDER}")
    logger.info(f"🔌 Backend URL: {BACKEND_URL}")
    logger.info(f"🎯 Source ID: {SOURCE_ID}")
    
    # Check backend health
    logger.info("🔍 Checking backend health...")
    if not check_backend_health():
        logger.warning("⚠️  Backend is offline. Will retry uploads when it comes online.")
    
    # Start watching
    event_handler = ImageUploadHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=False)
    
    try:
        observer.start()
        logger.info("👀 Watching for new images...")
        logger.info("Press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
        observer.stop()
    
    finally:
        observer.join()
        logger.info("✅ Watcher stopped")


if __name__ == "__main__":
    main()
