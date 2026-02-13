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