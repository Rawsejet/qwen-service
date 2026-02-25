#!/bin/bash
# Check Qwen3-Coder-Next server status

PID_FILE=~/qwen-service/qwen-server.pid

if [ ! -f "$PID_FILE" ]; then
    echo "Status: NOT RUNNING (no PID file)"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p $PID > /dev/null 2>&1; then
    echo "Status: NOT RUNNING (stale PID file)"
    rm "$PID_FILE"
    exit 1
fi

echo "Status: RUNNING (PID $PID)"

# Check if server is responding
if curl -s http://127.0.0.1:8085/health > /dev/null 2>&1; then
    echo "Health: OK"
else
    echo "Health: NOT RESPONDING"
fi

# Show VRAM usage
echo ""
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
  awk '{printf "GPU %s (%s): %s / %s MB (%.1f%%)\n", $1, $2, $3, $4, ($3/$4)*100}'