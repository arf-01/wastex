# Troubleshooting Guide

## Common Issues

### Issue 1: "Backend is offline"

**Symptoms**:
```
⚠️  Backend health check failed: ...
⚠️  Backend is offline. Will retry uploads when it comes online.
```

**Solutions**:

1. Check backend is running:
   ```powershell
   # On your Windows machine
   python manage.py runserver 0.0.0.0:8000
   ```

2. Verify backend IP:
   ```powershell
   ipconfig  # Get your IPv4 Address
   ```

3. Check firewall allows port 8000:
   ```powershell
   netstat -an | findstr :8000
   ```

5. Test from Pi:
   ```bash
   ping 192.169.0.120  # Your Raspberry Pi IP
   curl http://192.169.0.120:8000/api/pi/health/
   ```

5. Update BACKEND_URL in script with correct IP

---

### Issue 2: "Connection refused"

**Symptoms**:
```
⚠️  Connection failed (attempt 1/3)
```

**Solutions**:

1. Verify BACKEND_URL in script:
   ```bash
   grep "BACKEND_URL" ~/image_watcher.py
   ```

2. Check backend is listening:
   ```bash
   # From Pi, test connection to backend
   nc -zv 192.169.0.111 8000
   ```

3. Check network connectivity:
   ```bash
   ping 8.8.8.8  # Google DNS
   ```

4. Update script with correct IP and restart

---

### Issue 3: Images not uploading

**Symptoms**:
- No "New image detected" messages
- No output in logs

**Solutions**:

1. Check images are being created:
   ```bash
   ls -la ~/webcam_captures/
   ```

2. Check folder permissions:
   ```bash
   chmod 755 ~/webcam_captures
   ```

3. Check watcher is running:
   ```bash
   ps aux | grep image_watcher
   ```

4. Check logs:
   ```bash
   tail -50 /tmp/pi_image_watcher_pi_001.log
   ```

5. Test with manual image:
   ```bash
   python3 << 'EOF'
   from PIL import Image
   img = Image.new('RGB', (100, 100), color='blue')
   img.save('/home/dhruba001/webcam_captures/manual_test.jpg')
   EOF
   ```

---

### Issue 4: "No module named watchdog"

**Symptoms**:
```
ModuleNotFoundError: No module named 'watchdog'
```

**Solutions**:

```bash
pip install --upgrade watchdog requests pillow
```

Verify installation:
```bash
python3 -c "import watchdog; print(watchdog.__version__)"
```

---

### Issue 5: "No module named requests"

**Symptoms**:
```
ModuleNotFoundError: No module named 'requests'
```

**Solutions**:

```bash
pip install requests
```

---

### Issue 6: Backend receives images but no results

**Symptoms**:
- Images uploaded successfully (✅ message)
- But no inference results in logs

**Solutions**:

1. Check Django migration applied:
   ```powershell
   python manage.py migrate
   ```

2. Check TensorFlow installed:
   ```powershell
   python -c "import tensorflow; print(tensorflow.__version__)"
   ```

3. Check model file exists:
   ```powershell
   dir models\logits_mdl.keras
   ```

4. Check backend logs:
   ```powershell
   # Look at Django console output while running
   ```

5. Test API manually:
   ```powershell
   python test_pi_api.py http://localhost:8000 sample_image.jpg
   ```

---

### Issue 7: High CPU usage

**Symptoms**:
- Watcher consuming 50%+ CPU

**Solutions**:

1. Check if watching too large folder
2. Reduce retry attempts:
   ```python
   MAX_RETRIES = 1  # Reduce from 3
   ```

3. Increase sleep interval:
   ```python
   time.sleep(2)  # In on_created method
   ```

---

### Issue 8: Systemd service not starting

**Symptoms**:
```
sudo systemctl status pi-image-watcher
● pi-image-watcher.service - Disabled
```

**Solutions**:

1. Check service file exists:
   ```bash
   ls -la /etc/systemd/system/pi-image-watcher.service
   ```

2. Reload systemd:
   ```bash
   sudo systemctl daemon-reload
   ```

3. Check service syntax:
   ```bash
   systemd-analyze verify /etc/systemd/system/pi-image-watcher.service
   ```

4. View service logs:
   ```bash
   sudo journalctl -u pi-image-watcher -n 50
   ```

5. Manually run script to debug:
   ```bash
   python3 ~/image_watcher.py
   ```

---

## Debugging Commands

### Check Script Configuration

```bash
# View current settings
head -50 ~/image_watcher.py | grep -E "^(WATCH_FOLDER|BACKEND_URL|SOURCE_ID)"

# Change BACKEND_URL
sed -i 's/BACKEND_URL = .*/BACKEND_URL = "http:\/\/192.169.0.111:8000"/g' ~/image_watcher.py
```

### Monitor in Real-Time

```bash
# Watch logs while creating images
tail -f /tmp/pi_image_watcher_pi_001.log &
python3 << 'EOF'
from PIL import Image
import time
for i in range(5):
    img = Image.new('RGB', (100, 100), color='red')
    img.save(f'/home/dhruba001/webcam_captures/test_{i}.jpg')
    time.sleep(2)
EOF
```

### Test Network

```bash
# Check if backend reachable
curl -v http://192.169.0.111:8000/api/pi/health/

# Check DNS resolution
nslookup example.com

# Note: Windows backend is at 192.169.0.111:8000 on your WiFi
```

### Test File System

```bash
# Check folder permissions
ls -la ~/webcam_captures/

# Check disk space
df -h ~

# Check file creation
inotifywait -m ~/webcam_captures/
```

---

## Getting Help

If you're stuck:

1. Check the logs: `tail -f /tmp/pi_image_watcher_pi_001.log`
2. Check network: Test connection to your backend
3. Check backend: Visit `http://127.0.0.1:8000` (or your backend IP) in browser
4. Check service: `sudo systemctl status pi-image-watcher`
5. Check script config: `head -20 ~/image_watcher.py` and verify BACKEND_URL

Then fix the issue step by step!

---

## Quick Fix Checklist

- [ ] Backend running on correct port
- [ ] BACKEND_URL has correct IP
- [ ] Watch folder exists and has permissions
- [ ] Dependencies installed: `pip install watchdog requests`
- [ ] Script is executable: `chmod +x ~/image_watcher.py`
- [ ] Network can reach backend: `ping backend-ip`
- [ ] Log file writable: `touch /tmp/test_log.txt`
- [ ] Django migration applied on backend: `python manage.py migrate`

Fix any ✗ items and try again!
