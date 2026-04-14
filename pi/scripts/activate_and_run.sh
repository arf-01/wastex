#!/bin/bash
# Activate virtual environment and run image_watcher.py
# Usage: ./activate_and_run.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Image Watcher Launcher${NC}"
echo -e "${BLUE}================================${NC}"

# Check if venv exists
if [ ! -d ~/wastex_venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv ~/wastex_venv
fi

# Activate venv
echo "Activating virtual environment..."
source ~/wastex_venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo "Installing dependencies..."
pip install watchdog requests pillow -q

echo -e "${GREEN}✅ Environment ready!${NC}"
echo ""

# Run the watcher
echo "Starting image watcher..."
python3 ~/image_watcher.py
