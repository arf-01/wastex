#!/usr/bin/env python3
"""
Raspberry Pi Real-time Image Upload Service

Monitors a folder for new images and uploads them to the WasteX
backend for inference using DRF token authentication.

Config file: /etc/wastex/config.ini  (see template below)

    [server]
    BACKEND_URL = http://192.169.0.111:8000

    [identity]
    BIN_ID     = bin_cafeteria
    USER_TOKEN = <paste token from WasteX edge profile page>

Setup:
    1. Copy this script to your Raspberry Pi
    2. Create the config file at /etc/wastex/config.ini
    3. Install deps:  pip install watchdog requests
    4. Run:          python3 image_watcher.py
    5. Or install as a service: see /pi/service/wastex-watcher.service
"""

import os
import sys
import time
import logging
import configparser
import requests
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION  (loaded from /etc/wastex/config.ini)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONFIG_FILE = "/etc/wastex/config.ini"

def load_config():
    """Load and validate configuration from the config file."""
    cfg = configparser.ConfigParser()

    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Config file not found: {CONFIG_FILE}")
        print()
        print("   Create it with the following contents:")
        print()
        print("   [server]")
        print("   BACKEND_URL = http://<your-backend-ip>:8000")
        print()
        print("   [identity]")
        print("   BIN_ID     = bin_<location>")
        print("   USER_TOKEN = <paste from WasteX Edge → API Key page>")
        sys.exit(1)

    cfg.read(CONFIG_FILE)

    # Required values
    try:
        backend_url = cfg["server"]["BACKEND_URL"].rstrip("/")
        bin_id      = cfg["identity"]["BIN_ID"]
        user_token  = cfg["identity"]["USER_TOKEN"]
    except KeyError as e:
        print(f"❌ Missing config key: {e}")
        print(f"   Check {CONFIG_FILE} and ensure all keys are present.")
        sys.exit(1)

    # Optional values with defaults
    watch_folder   = cfg.get("watcher", "WATCH_FOLDER",  fallback="/home/pi/webcam_captures")
    max_retries    = int(cfg.get("watcher", "MAX_RETRIES",   fallback="3"))
    retry_delay    = float(cfg.get("watcher", "RETRY_DELAY",  fallback="2.0"))
    valid_exts     = {e.strip().lower() for e in
                      cfg.get("watcher", "VALID_EXTENSIONS",
                               fallback=".jpg,.jpeg,.png,.bmp").split(",")}

    return {
        "backend_url":   backend_url,
        "bin_id":        bin_id,
        "user_token":    user_token,
        "watch_folder":  watch_folder,
        "max_retries":   max_retries,
        "retry_delay":   retry_delay,
        "valid_exts":    valid_exts,
    }


# Load config at module level so handler can access it
CONFIG = load_config()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOG_FILE = f"/tmp/wastex_watcher_{CONFIG['bin_id']}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTH HEADER HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def auth_headers() -> dict:
    """Return the DRF token auth header for this Pi's edge account."""
    return {"Authorization": f"Token {CONFIG['user_token']}"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE WATCHER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ImageUploadHandler(FileSystemEventHandler):
    """Watch a folder and upload new images to the WasteX backend."""

    def on_created(self, event):
        """Triggered when a new file is created in the watched folder."""
        if event.is_directory:
            return

        file_path = event.src_path
        if Path(file_path).suffix.lower() not in CONFIG["valid_exts"]:
            return

        # Small delay — ensures the file is fully written before we open it
        time.sleep(0.5)

        logger.info(f"📷 New image detected: {file_path}")
        self.upload_image(file_path)

    def upload_image(self, file_path: str):
        """Upload image to backend with token auth, retrying on failure."""
        if not os.path.exists(file_path):
            logger.warning(f"File vanished before upload: {file_path}")
            return

        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                with open(file_path, "rb") as f:
                    response = requests.post(
                        f"{CONFIG['backend_url']}/classifier/api/pi/inference/",
                        headers=auth_headers(),
                        files={"image": f},
                        data={"source": CONFIG["bin_id"]},
                        timeout=30,
                    )

                if response.status_code == 200:
                    result = response.json()
                    logger.info("✅ Upload successful!")
                    logger.info(f"   Bin:             {CONFIG['bin_id']}")
                    logger.info(f"   Predicted class: {result.get('predicted_class')}")
                    logger.info(f"   Saved to DB:     {result.get('saved_to_db')}")
                    return

                elif response.status_code == 401:
                    logger.error("🔒 Authentication failed (401) — check USER_TOKEN in config.ini")
                    logger.error("   Generate a fresh token at:  <backend>/classifier/edge/api-key/")
                    return  # No point retrying — auth is wrong

                elif response.status_code == 403:
                    logger.error("🚫 Permission denied (403) — account may not be in EdgeUsers group")
                    return

                else:
                    logger.error(f"❌ Backend error (attempt {attempt}/{CONFIG['max_retries']})")
                    logger.error(f"   Status:   {response.status_code}")
                    logger.error(f"   Response: {response.text[:300]}")

            except requests.exceptions.ConnectionError:
                logger.error(f"⚠️  Connection failed (attempt {attempt}/{CONFIG['max_retries']})")

            except Exception as e:
                logger.error(f"❌ Unexpected error (attempt {attempt}/{CONFIG['max_retries']}): {e}")

            if attempt < CONFIG["max_retries"]:
                time.sleep(CONFIG["retry_delay"])

        logger.error(f"💥 Giving up after {CONFIG['max_retries']} attempts: {file_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEALTH CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_backend_health() -> bool:
    """Verify the backend is online and register bin status."""
    try:
        r = requests.get(
            f"{CONFIG['backend_url']}/classifier/api/pi/health/?source={CONFIG['bin_id']}",
            headers=auth_headers(),
            timeout=5,
        )
        if r.status_code == 200:
            logger.info("✅ Backend is online!")
            return True
    except Exception as e:
        logger.warning(f"⚠️  Health check failed: {e}")
    return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Start the file watcher service."""

    logger.info("=" * 70)
    logger.info("🚀 WasteX Pi Image Watcher Service")
    logger.info("=" * 70)
    logger.info(f"   Config:       {CONFIG_FILE}")
    logger.info(f"   Backend:      {CONFIG['backend_url']}")
    logger.info(f"   Bin ID:       {CONFIG['bin_id']}")
    logger.info(f"   Watch folder: {CONFIG['watch_folder']}")
    logger.info(f"   Auth token:   {CONFIG['user_token'][:8]}…  (truncated)")

    # Create watch folder if needed
    watch_path = Path(CONFIG["watch_folder"])
    watch_path.mkdir(parents=True, exist_ok=True)

    # Health check
    logger.info("🔍 Checking backend…")
    if not check_backend_health():
        logger.warning("⚠️  Backend offline — will retry uploads when it comes back.")

    # Start watching
    handler  = ImageUploadHandler()
    observer = Observer()
    observer.schedule(handler, watch_path, recursive=False)

    try:
        observer.start()
        logger.info("👀 Watching for new images… (Ctrl+C to stop)")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down…")
        observer.stop()

    finally:
        observer.join()
        logger.info("✅ Watcher stopped cleanly")


if __name__ == "__main__":
    main()
