#!/bin/bash
# View Qwen3-Coder-Next server logs (works for both vLLM and llama.cpp)

LOG_FILE=~/qwen-service/qwen-server.log
BACKEND_FILE=~/qwen-service/qwen-server.backend

if [ ! -f "$LOG_FILE" ]; then
    echo "No log file found. Has the server been started yet?"
    exit 1
fi

BACKEND=$(cat "$BACKEND_FILE" 2>/dev/null | awk '{print $2}')
[ -n "$BACKEND" ] && echo "Backend: $BACKEND" && echo ""

if [ "$1" == "-f" ]; then
    tail -f "$LOG_FILE"
else
    tail -n 50 "$LOG_FILE"
fi
