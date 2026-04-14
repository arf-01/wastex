# Raspberry Pi Real-Time Image Processing

This folder contains all Pi-related code and configuration for the real-time image processing system.

## Folder Structure

```
pi/
├── scripts/
│   └── image_watcher.py          ← Main watcher script for Pi
├── systemd/
│   └── pi-image-watcher.service  ← Systemd service file
└── docs/
    ├── SETUP.md                  ← Setup instructions
    ├── TROUBLESHOOTING.md        ← Troubleshooting guide
    └── API_REFERENCE.md          ← API endpoints reference
```

## Quick Start

### 1. Copy Script to Pi

```powershell
scp pi/scripts/image_watcher.py dhruba001@192.169.0.120:~/
```

### 2. Install Dependencies

```bash
ssh dhruba001@192.169.0.120
pip install watchdog requests pillow
mkdir -p ~/webcam_captures
```

### 3. Configure

Edit `~/image_watcher.py`:
```python
BACKEND_URL = "http://YOUR_IP:8000"  # Your laptop IP
WATCH_FOLDER = "/home/dhruba001/webcam_captures"
SOURCE_ID = "pi_001"
```

### 4. Run

```bash
python3 ~/image_watcher.py
```

## Setup on Auto-Start

Copy the systemd service file:

```bash
sudo cp pi/systemd/pi-image-watcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pi-image-watcher
sudo systemctl start pi-image-watcher
```

## Documentation

- **[SETUP.md](docs/SETUP.md)** - Complete setup guide
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Backend API endpoints

## Dependencies

```bash
pip install watchdog requests pillow
```

- `watchdog` - File system monitoring
- `requests` - HTTP client
- `pillow` - Image handling (optional)

## Configuration Variables

In `image_watcher.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `WATCH_FOLDER` | `/home/dhruba001/webcam_captures` | Folder to monitor |
| `BACKEND_URL` | `http://192.168.x.x:8000` | Backend server address |
| `SOURCE_ID` | `pi_001` | Device identifier |
| `MAX_RETRIES` | `3` | Upload retry attempts |
| `RETRY_DELAY` | `2` | Retry delay in seconds |

## How It Works

1. **Monitor**: Watches folder for new images
2. **Detect**: Uses inotify to detect file creation
3. **Upload**: Sends image to backend via HTTP
4. **Retry**: Auto-retries on failure (3 attempts)
5. **Log**: Logs all activity to `/tmp/pi_image_watcher_pi_001.log`

## Integration with Your Webcam Script

Your webcam capture script should save images to:
```python
CAPTURE_FOLDER = "/home/dhruba001/webcam_captures"
output_path = os.path.join(CAPTURE_FOLDER, f"capture_{timestamp}.jpg")
```

The watcher will automatically detect and upload them!

## Monitoring

### View Live Logs

```bash
tail -f /tmp/pi_image_watcher_pi_001.log
```

### Check Service Status

```bash
sudo systemctl status pi-image-watcher
journalctl -u pi-image-watcher -f
```

## Troubleshooting

For common issues, see **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**

Key issues:
- Backend offline
- Connection refused
- Images not detected
- Module not found

## Support

See the `docs/` folder for detailed guides.
