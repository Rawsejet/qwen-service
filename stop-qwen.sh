#!/bin/bash
# Stop Qwen3 model servers (Coder on 8085, VL on 8086)
# Usage: ./stop-qwen.sh          → interactive choice
#        ./stop-qwen.sh coder    → stop Coder only
#        ./stop-qwen.sh vl       → stop VL only
#        ./stop-qwen.sh all      → stop everything

PID_FILE_CODER=~/qwen-service/qwen-server.pid
PID_FILE_VL=~/qwen-service/qwen-vl-server.pid
BACKEND_FILE=~/qwen-service/qwen-server.backend

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
kill_vllm_by_port() {
    local PORT=$1
    local PID_FILE=$2
    local LABEL=$3
    echo "Stopping $LABEL (port $PORT)..."

    # Kill by process group if we have the PID
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        PGID=$(ps -o pgid= -p $PID 2>/dev/null | tr -d ' ')
        if [ -n "$PGID" ] && [ "$PGID" != "0" ]; then
            echo "  Killing process group $PGID..."
            kill -- -$PGID 2>/dev/null
        fi
    fi

    # Sweep by port binding
    fuser -k ${PORT}/tcp 2>/dev/null

    sleep 3

    # Force-kill any stragglers bound to that port
    fuser -9k ${PORT}/tcp 2>/dev/null
}

kill_all_vllm() {
    echo "Sweeping all vLLM processes..."
    pkill -f "vllm serve" 2>/dev/null
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "from vllm" 2>/dev/null
    sleep 3
    pkill -9 -f "vllm serve" 2>/dev/null
    pkill -9 -f "vllm.entrypoints" 2>/dev/null
    pkill -9 -f "from vllm" 2>/dev/null
    sleep 2
}

kill_llama_cpp() {
    local PID_FILE=$1
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Stopping llama.cpp server (PID $PID)..."
            kill $PID
            for i in {1..15}; do
                if ! ps -p $PID > /dev/null 2>&1; then
                    echo "✓ Stopped."
                    break
                fi
                sleep 1
            done
            if ps -p $PID > /dev/null 2>&1; then
                echo "Force stopping..."
                kill -9 $PID 2>/dev/null
                echo "✓ Stopped (forced)."
            fi
        else
            echo "llama.cpp not running (stale PID file)."
        fi
    else
        pkill -f "llama-server" 2>/dev/null && echo "Killed llama-server." || echo "No llama-server found."
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

stop_coder() {
    BACKEND="unknown"
    [ -f "$BACKEND_FILE" ] && BACKEND=$(cat "$BACKEND_FILE")
    case "$BACKEND" in
        vllm)
            kill_vllm_by_port 8085 "$PID_FILE_CODER" "Qwen3-Coder (vLLM)"
            ;;
        llama.cpp)
            kill_llama_cpp "$PID_FILE_CODER"
            ;;
        *)
            echo "Unknown backend for Coder — sweeping port 8085..."
            kill_vllm_by_port 8085 "$PID_FILE_CODER" "Qwen3-Coder"
            pkill -f "llama-server" 2>/dev/null
            ;;
    esac
    rm -f "$PID_FILE_CODER" "$BACKEND_FILE"
    echo "✓ Coder server stopped."
}

stop_vl() {
    if [ -f "$PID_FILE_VL" ]; then
        kill_vllm_by_port 8086 "$PID_FILE_VL" "Qwen3-VL (vLLM)"
    else
        echo "No VL PID file — sweeping port 8086..."
        fuser -9k 8086/tcp 2>/dev/null
    fi
    rm -f "$PID_FILE_VL"
    echo "✓ VL server stopped."
}

show_gpu_status() {
    echo ""
    echo "GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU %s: %s MB used, %s MB free\n", $1, $3, $4}'
    echo ""
}

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
TARGET="${1:-}"

if [ -z "$TARGET" ]; then
    # Detect what's running
    CODER_RUNNING=false
    VL_RUNNING=false
    [ -f "$PID_FILE_CODER" ] && ps -p $(cat "$PID_FILE_CODER") > /dev/null 2>&1 && CODER_RUNNING=true
    [ -f "$PID_FILE_VL" ] && ps -p $(cat "$PID_FILE_VL") > /dev/null 2>&1 && VL_RUNNING=true

    echo "========================================="
    echo "  Stop Qwen3 Servers"
    echo "========================================="
    echo ""
    echo "Running:"
    $CODER_RUNNING && echo "  ✓ Qwen3-Coder (port 8085)" || echo "  ✗ Qwen3-Coder (not running)"
    $VL_RUNNING && echo "  ✓ Qwen3-VL    (port 8086)" || echo "  ✗ Qwen3-VL    (not running)"
    echo ""
    echo "1) Stop Coder only"
    echo "2) Stop VL only"
    echo "3) Stop all"
    echo ""
    read -p "Enter choice [1-3]: " choice
    case "$choice" in
        1) TARGET="coder" ;;
        2) TARGET="vl" ;;
        3) TARGET="all" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

case "$TARGET" in
    coder)
        stop_coder
        ;;
    vl)
        stop_vl
        ;;
    all)
        stop_coder
        stop_vl
        kill_all_vllm  # final sweep
        ;;
    *)
        echo "Usage: $0 [coder|vl|all]"
        exit 1
        ;;
esac

show_gpu_status
echo "Done."