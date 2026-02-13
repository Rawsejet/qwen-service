#!/bin/bash
# View Qwen3-Coder-Next server logs

LOG_FILE=~/qwen-service/qwen-server.log

if [ ! -f "$LOG_FILE" ]; then
    echo "No log file found"
    exit 1
fi

# Follow logs if -f flag provided
if [ "$1" == "-f" ]; then
    tail -f "$LOG_FILE"
else
    tail -n 50 "$LOG_FILE"
fi