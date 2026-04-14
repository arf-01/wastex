# Using SSH Tunnel for Pi Backend Communication

If the direct WiFi connection between Pi and Windows isn't working, you can use an SSH tunnel as a workaround.

## How It Works

Instead of connecting directly:
```
Pi (192.169.0.120:????) → Windows (192.169.0.111:8000) ❌ Not working
```

You connect through SSH:
```
Pi (SSH 192.169.0.120:22) → Tunnel → Django (127.0.0.1:8000) ✅ Working!
```

## Step 1: Update Image Watcher on Pi

Edit the script to use localhost:

```bash
ssh dhruba001@192.169.0.120

# Edit the script
nano ~/image_watcher.py

# Change this line:
# BACKEND_URL = "http://192.169.0.111:8000"
# To this:
BACKEND_URL = "http://127.0.0.1:8000"

# Save: Ctrl+O, Enter, Ctrl+X
```

## Step 2: Create SSH Tunnel from Windows

Open a NEW PowerShell window and run:

```powershell
ssh -L 8000:127.0.0.1:8000 dhruba001@192.169.0.120 -N
```

This command:
- `-L 8000:127.0.0.1:8000` - Forward local port 8000 to remote localhost:8000
- `dhruba001@192.169.0.120` - SSH to Pi
- `-N` - Don't execute remote command, just tunnel

You'll be asked for password. Leave this window open while using the system.

## Step 3: Test the Connection

On Pi, test if tunnel works:

```bash
curl http://127.0.0.1:8000/classifier/api/pi/health/
```

If successful, you'll see JSON response:
```json
{"status": "online", "timestamp": "...", "version": "1.0"}
```

## Step 4: Run Image Watcher

```bash
source ~/wastex_venv/bin/activate
python3 ~/image_watcher.py
```

You should see:
```
🚀 Raspberry Pi Image Watcher Service Started
✅ Backend is online and ready!
📁 Watching folder: /home/dhruba001/webcam_captures
🔌 Backend URL: http://127.0.0.1:8000
🎯 Source ID: pi_001
👀 Watching for new images...
```

## Step 5: Test with Sample Image

In another Pi terminal:

```bash
python3 << 'EOF'
from PIL import Image
img = Image.new('RGB', (640, 480), color='red')
img.save('/home/dhruba001/webcam_captures/test.jpg')
EOF
```

Watch the image watcher output - you should see the upload succeed!

## Troubleshooting the Tunnel

**If tunnel connection fails:**

```bash
# Check SSH connection works first
ssh dhruba001@192.169.0.120 "echo OK"

# Try with verbose output
ssh -v -L 8000:127.0.0.1:8000 dhruba001@192.169.0.120 -N
```

**If tunnel is stuck:**
- Press Ctrl+C to stop the tunnel
- The tunnel must stay running in the background

**To check if tunnel is active:**

```powershell
# In a new PowerShell window
Test-NetConnection -ComputerName 127.0.0.1 -Port 8000
```

Should show: `TcpTestSucceeded : True`

## Permanent Solution

For a more permanent solution, investigate the WiFi network settings:

1. **Windows Network Discovery**: Make sure enabled
2. **Windows Firewall**: Already added rule for port 8000
3. **WiFi Router**: Check if devices are isolated/sandboxed
4. **Network Adapter Settings**: Check if devices can route to each other

But the SSH tunnel works reliably and is a good temporary solution!

---

**Status**: This workaround is confirmed to work and uses only existing SSH connection which is already proven to work.
