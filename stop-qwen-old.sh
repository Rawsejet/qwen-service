#!/bin/bash
# Stop Qwen3-Coder-Next server

PID_FILE=~/qwen-service/qwen-server.pid

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Server may not be running."
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p $PID > /dev/null 2>&1; then
    echo "Server not running (stale PID file)"
    rm "$PID_FILE"
    exit 1
fi

echo "Stopping server (PID $PID)..."
kill $PID

# Wait for graceful shutdown
for i in {1..10}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "Server stopped successfully"
        rm "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "Forcing shutdown..."
kill -9 $PID
rm "$PID_FILE"
echo "Server stopped (forced)"

# Shrink swap back to 8GB if expanded
echo ""
echo "Checking swap size..."
CURRENT_SWAP=$(grep "^/swap.img" /proc/swaps | awk '{print $3}')
if [ -z "$CURRENT_SWAP" ]; then
    CURRENT_SWAP=0
fi
CURRENT_SWAP_GB=$((CURRENT_SWAP / 1048576))

if [ "$CURRENT_SWAP_GB" -gt 8 ]; then
    echo "↓ Shrinking swap from ${CURRENT_SWAP_GB}GB back to 8GB..."

    # Disable current swap, create new 8GB one
    sudo swapoff /swap.img 2>/dev/null
    sudo rm /swap.img 2>/dev/null
    sudo fallocate -l 8G /swap.img 2>/dev/null
    sudo chmod 600 /swap.img 2>/dev/null
    sudo mkswap /swap.img 2>/dev/null
    sudo swapon /swap.img 2>/dev/null

    echo "✓ Swap shrunk to 8GB"
fi