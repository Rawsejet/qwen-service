#!/bin/bash
# Stop Qwen3-Coder-Next server (vLLM or llama.cpp)

PID_FILE=~/qwen-service/qwen-server.pid
BACKEND_FILE=~/qwen-service/qwen-server.backend
LOG_FILE=~/qwen-service/qwen-server.log

BACKEND="unknown"
if [ -f "$BACKEND_FILE" ]; then
    BACKEND=$(cat "$BACKEND_FILE")
fi

echo "Backend: $BACKEND"

# ─────────────────────────────────────────────
# Kill all matching processes by name
# vLLM spawns multiple workers that won't die
# from killing just the parent PID
# ─────────────────────────────────────────────
kill_all_vllm() {
    echo "Killing all vLLM processes..."
    # Kill by process group if we have the PID
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        PGID=$(ps -o pgid= -p $PID 2>/dev/null | tr -d ' ')
        if [ -n "$PGID" ] && [ "$PGID" != "0" ]; then
            echo "  Killing process group $PGID..."
            kill -- -$PGID 2>/dev/null
        fi
    fi
    # Also sweep any stragglers by name
    pkill -f "vllm serve" 2>/dev/null
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "from vllm" 2>/dev/null
    # Give workers a moment to exit
    sleep 3
    # Force-kill anything still alive
    pkill -9 -f "vllm serve" 2>/dev/null
    pkill -9 -f "vllm.entrypoints" 2>/dev/null
    pkill -9 -f "from vllm" 2>/dev/null
    # Wait for GPU VRAM to clear
    sleep 2
}

kill_llama_cpp() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Stopping llama.cpp server (PID $PID)..."
            kill $PID
            # Wait up to 15s for graceful exit
            for i in {1..15}; do
                if ! ps -p $PID > /dev/null 2>&1; then
                    echo "✓ Server stopped."
                    break
                fi
                sleep 1
            done
            # Force kill if still alive
            if ps -p $PID > /dev/null 2>&1; then
                echo "Force stopping..."
                kill -9 $PID 2>/dev/null
                echo "✓ Server stopped (forced)."
            fi
        else
            echo "Server not running (stale PID file)."
        fi
    else
        echo "No PID file found. Sweeping by name..."
        pkill -f "llama-server" 2>/dev/null && echo "Killed llama-server process." || echo "No llama-server found."
    fi

    # Shrink swap back if it was expanded
    CURRENT_SWAP=$(grep "^/swap.img" /proc/swaps | awk '{print $3}')
    CURRENT_SWAP=${CURRENT_SWAP:-0}
    CURRENT_SWAP_GB=$((CURRENT_SWAP / 1048576))
    if [ "$CURRENT_SWAP_GB" -gt 10 ]; then
        echo "Shrinking swap back to 8GB..."
        sudo fallocate -l 8G /swap_new.img 2>/dev/null
        sudo chmod 600 /swap_new.img 2>/dev/null
        sudo mkswap /swap_new.img 2>/dev/null
        sudo swapon /swap_new.img 2>/dev/null
        sudo swapoff /swap.img 2>/dev/null
        sudo rm /swap.img 2>/dev/null
        sudo mv /swap_new.img /swap.img 2>/dev/null
        echo "✓ Swap restored to 8GB."
    fi
}

# Route to correct killer
case "$BACKEND" in
    vllm)
        kill_all_vllm
        ;;
    llama.cpp)
        kill_llama_cpp
        ;;
    *)
        echo "Unknown or missing backend — killing everything that looks like either server..."
        kill_all_vllm
        pkill -f "llama-server" 2>/dev/null
        ;;
esac

# Cleanup state files
rm -f "$PID_FILE" "$BACKEND_FILE"

# Confirm GPU VRAM cleared
echo ""
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader,nounits | \
    awk -F',' '{printf "  GPU %s: %s MB used, %s MB free\n", $1, $3, $4}'
echo ""
echo "Done."