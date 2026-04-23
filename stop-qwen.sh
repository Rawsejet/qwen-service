#!/bin/bash
# Stop Qwen3 model servers
# Usage: ./stop-qwen.sh              → interactive choice
#        ./stop-qwen.sh coder        → stop Coder only
#        ./stop-qwen.sh vl           → stop VL-32B only
#        ./stop-qwen.sh vl4b         → stop VL-4B only
#        ./stop-qwen.sh 27b          → stop 27B only
#        ./stop-qwen.sh 122b         → stop 122B only
#        ./stop-qwen.sh omnicoder    → stop OmniCoder-9B only
#        ./stop-qwen.sh agg35b       → stop Aggressive-35B only
#        ./stop-qwen.sh agg27b       → stop Aggressive-27B only
#        ./stop-qwen.sh q36-35b      → stop Qwen3.6-35B-A3B only
#        ./stop-qwen.sh q36-27b      → stop Qwen3.6-27B only
#        ./stop-qwen.sh all          → stop everything

PID_FILE_CODER=~/qwen-service/qwen-server.pid
PID_FILE_VL=~/qwen-service/qwen-vl-server.pid
PID_FILE_VL4B=~/qwen-service/qwen-vl4b-server.pid
PID_FILE_27B=~/qwen-service/qwen-27b-server.pid
PID_FILE_122B=~/qwen-service/qwen-122b-server.pid
PID_FILE_OMNICODER=~/qwen-service/omnicoder-9b-server.pid
PID_FILE_AGG35B=~/qwen-service/aggressive-35b-server.pid
PID_FILE_AGG27B=~/qwen-service/aggressive-27b-server.pid
PID_FILE_Q36_35B=~/qwen-service/qwen36-35b-server.pid
PID_FILE_Q36_27B=~/qwen-service/qwen36-27b-server.pid
BACKEND_FILE=~/qwen-service/qwen-server.backend
BACKEND_FILE_27B=~/qwen-service/qwen-27b-server.backend
BACKEND_FILE_122B=~/qwen-service/qwen-122b-server.backend

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

    # Kill by port for all known model ports
    for PORT in 8085 8086 8087 8088 8089 8090 8091 8092 8093 8094; do
        fuser -k ${PORT}/tcp 2>/dev/null
    done

    # Kill by cmdline patterns (catches the main serve process and entrypoint workers)
    pkill -f "vllm serve" 2>/dev/null
    pkill -f "vllm.entrypoints" 2>/dev/null
    pkill -f "vllm.worker" 2>/dev/null
    pkill -f "vllm.executor" 2>/dev/null
    pkill -f "from vllm" 2>/dev/null

    # Kill by process name (vLLM renames its workers to VLLM::EngineCore, VLLM::Worker_TP*, etc.)
    pkill "VLLM" 2>/dev/null

    sleep 3

    # Force-kill any survivors
    for PORT in 8085 8086 8087 8088 8089 8090 8091 8092 8093 8094; do
        fuser -9k ${PORT}/tcp 2>/dev/null
    done
    pkill -9 -f "vllm serve" 2>/dev/null
    pkill -9 -f "vllm.entrypoints" 2>/dev/null
    pkill -9 -f "vllm.worker" 2>/dev/null
    pkill -9 -f "vllm.executor" 2>/dev/null
    pkill -9 -f "from vllm" 2>/dev/null
    pkill -9 "VLLM" 2>/dev/null

    sleep 2
}

kill_sglang() {
    local PID_FILE=$1
    local PORT=$2
    local LABEL=${3:-"SGLang server"}
    echo "Stopping $LABEL (port $PORT)..."

    # Kill by PID file first (gets the main process)
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "  Killing main process (PID $PID)..."
            kill $PID 2>/dev/null
            for i in {1..10}; do
                if ! ps -p $PID > /dev/null 2>&1; then break; fi
                sleep 1
            done
            # Force kill if still alive
            if ps -p $PID > /dev/null 2>&1; then
                echo "  Force killing PID $PID..."
                kill -9 $PID 2>/dev/null
            fi
        else
            echo "  Main process not running (stale PID file)."
        fi
    fi

    # Kill sglang child processes (schedulers, detokenizer, compile workers)
    pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
    pkill -f "sglang::scheduler" 2>/dev/null
    pkill -f "sglang::detokenizer" 2>/dev/null
    pkill -f "torch._inductor.compile_worker" 2>/dev/null
    sleep 2

    # Force kill any remaining sglang processes
    pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null
    pkill -9 -f "sglang::scheduler" 2>/dev/null
    pkill -9 -f "sglang::detokenizer" 2>/dev/null
    pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null

    # Final sweep: kill anything on the port
    fuser -9k ${PORT}/tcp 2>/dev/null

    sleep 1

    # Verify
    if ss -tlnp | grep -q ":${PORT} " 2>/dev/null; then
        echo "  ⚠ Warning: port $PORT still in use!"
    else
        echo "  ✓ Port $PORT is free."
    fi
}

kill_all_sglang() {
    echo "Sweeping all SGLang processes..."
    pkill -f "sglang.launch_server" 2>/dev/null
    pkill -f "sglang::scheduler" 2>/dev/null
    pkill -f "sglang::detokenizer" 2>/dev/null
    pkill -f "torch._inductor.compile_worker" 2>/dev/null
    sleep 3
    pkill -9 -f "sglang.launch_server" 2>/dev/null
    pkill -9 -f "sglang::scheduler" 2>/dev/null
    pkill -9 -f "sglang::detokenizer" 2>/dev/null
    pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null
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
        sglang)
            kill_sglang "$PID_FILE_CODER" 8085 "Qwen3-Coder (SGLang)"
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
        kill_vllm_by_port 8086 "$PID_FILE_VL" "Qwen3-VL-32B (vLLM)"
    else
        echo "No VL-32B PID file — sweeping port 8086..."
        fuser -9k 8086/tcp 2>/dev/null
    fi
    rm -f "$PID_FILE_VL"
    echo "✓ VL-32B server stopped."
}

stop_vl4b() {
    if [ -f "$PID_FILE_VL4B" ]; then
        kill_vllm_by_port 8088 "$PID_FILE_VL4B" "Qwen3-VL-4B (vLLM)"
    else
        echo "No VL-4B PID file — sweeping port 8088..."
        fuser -9k 8088/tcp 2>/dev/null
    fi
    rm -f "$PID_FILE_VL4B"
    echo "✓ VL-4B server stopped."
}

stop_27b() {
    BACKEND="unknown"
    [ -f "$BACKEND_FILE_27B" ] && BACKEND=$(cat "$BACKEND_FILE_27B")
    case "$BACKEND" in
        vllm)
            kill_vllm_by_port 8087 "$PID_FILE_27B" "Qwen3.5-27B (vLLM)"
            ;;
        sglang)
            # Legacy: 27B was previously served via SGLang
            kill_sglang "$PID_FILE_27B" 8087 "Qwen3.5-27B (SGLang)"
            ;;
        *)
            echo "Unknown backend for 27B — sweeping port 8087..."
            kill_vllm_by_port 8087 "$PID_FILE_27B" "Qwen3.5-27B"
            kill_sglang "$PID_FILE_27B" 8087 "Qwen3.5-27B (stale SGLang sweep)"
            ;;
    esac
    rm -f "$PID_FILE_27B" "$BACKEND_FILE_27B"
    echo "✓ 27B server stopped."
}

stop_122b() {
    if [ -f "$PID_FILE_122B" ]; then
        kill_vllm_by_port 8089 "$PID_FILE_122B" "Qwen3.5-122B-A10B (vLLM)"
    else
        echo "No 122B PID file — sweeping port 8089..."
        fuser -9k 8089/tcp 2>/dev/null
    fi
    rm -f "$PID_FILE_122B" "$BACKEND_FILE_122B"
    echo "✓ 122B server stopped."
}

stop_omnicoder() {
    if [ -f "$PID_FILE_OMNICODER" ]; then
        kill_vllm_by_port 8090 "$PID_FILE_OMNICODER" "OmniCoder-9B (vLLM)"
    else
        echo "No OmniCoder-9B PID file — sweeping port 8090..."
        fuser -9k 8090/tcp 2>/dev/null
    fi
    rm -f "$PID_FILE_OMNICODER"
    echo "✓ OmniCoder-9B server stopped."
}

stop_agg35b() {
    kill_llama_cpp "$PID_FILE_AGG35B"
    fuser -9k 8091/tcp 2>/dev/null
    rm -f "$PID_FILE_AGG35B" ~/qwen-service/aggressive-35b-server.backend
    echo "✓ Aggressive-35B server stopped."
}

stop_agg27b() {
    kill_llama_cpp "$PID_FILE_AGG27B"
    fuser -9k 8092/tcp 2>/dev/null
    rm -f "$PID_FILE_AGG27B" ~/qwen-service/aggressive-27b-server.backend
    echo "✓ Aggressive-27B server stopped."
}

stop_q36_35b() {
    if [ -f "$PID_FILE_Q36_35B" ]; then
        kill_vllm_by_port 8093 "$PID_FILE_Q36_35B" "Qwen3.6-35B-A3B (vLLM)"
    else
        echo "No Qwen3.6-35B-A3B PID file — sweeping port 8093..."
        fuser -9k 8093/tcp 2>/dev/null
    fi
    rm -f "$PID_FILE_Q36_35B" ~/qwen-service/qwen36-35b-server.backend
    echo "✓ Qwen3.6-35B-A3B server stopped."
}

stop_q36_27b() {
    if [ -f "$PID_FILE_Q36_27B" ]; then
        kill_vllm_by_port 8094 "$PID_FILE_Q36_27B" "Qwen3.6-27B (vLLM)"
    else
        echo "No Qwen3.6-27B PID file — sweeping port 8094..."
        fuser -9k 8094/tcp 2>/dev/null
    fi
    rm -f "$PID_FILE_Q36_27B" ~/qwen-service/qwen36-27b-server.backend
    echo "✓ Qwen3.6-27B server stopped."
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
    VL4B_RUNNING=false
    B27_RUNNING=false
    B122_RUNNING=false
    OMNICODER_RUNNING=false
    AGG35B_RUNNING=false
    AGG27B_RUNNING=false
    Q36_35B_RUNNING=false
    Q36_27B_RUNNING=false
    [ -f "$PID_FILE_CODER" ] && ps -p $(cat "$PID_FILE_CODER") > /dev/null 2>&1 && CODER_RUNNING=true
    [ -f "$PID_FILE_VL" ] && ps -p $(cat "$PID_FILE_VL") > /dev/null 2>&1 && VL_RUNNING=true
    [ -f "$PID_FILE_VL4B" ] && ps -p $(cat "$PID_FILE_VL4B") > /dev/null 2>&1 && VL4B_RUNNING=true
    [ -f "$PID_FILE_27B" ] && ps -p $(cat "$PID_FILE_27B") > /dev/null 2>&1 && B27_RUNNING=true
    [ -f "$PID_FILE_122B" ] && ps -p $(cat "$PID_FILE_122B") > /dev/null 2>&1 && B122_RUNNING=true
    [ -f "$PID_FILE_OMNICODER" ] && ps -p $(cat "$PID_FILE_OMNICODER") > /dev/null 2>&1 && OMNICODER_RUNNING=true
    [ -f "$PID_FILE_AGG35B" ] && ps -p $(cat "$PID_FILE_AGG35B") > /dev/null 2>&1 && AGG35B_RUNNING=true
    [ -f "$PID_FILE_AGG27B" ] && ps -p $(cat "$PID_FILE_AGG27B") > /dev/null 2>&1 && AGG27B_RUNNING=true
    [ -f "$PID_FILE_Q36_35B" ] && ps -p $(cat "$PID_FILE_Q36_35B") > /dev/null 2>&1 && Q36_35B_RUNNING=true
    [ -f "$PID_FILE_Q36_27B" ] && ps -p $(cat "$PID_FILE_Q36_27B") > /dev/null 2>&1 && Q36_27B_RUNNING=true
    # Also detect by port if no PID file
    if ! $VL4B_RUNNING && ss -tlnp | grep -q ":8088 " 2>/dev/null; then VL4B_RUNNING=true; fi
    if ! $B27_RUNNING && ss -tlnp | grep -q ":8087 " 2>/dev/null; then B27_RUNNING=true; fi
    if ! $B122_RUNNING && ss -tlnp | grep -q ":8089 " 2>/dev/null; then B122_RUNNING=true; fi
    if ! $OMNICODER_RUNNING && ss -tlnp | grep -q ":8090 " 2>/dev/null; then OMNICODER_RUNNING=true; fi
    if ! $AGG35B_RUNNING && ss -tlnp | grep -q ":8091 " 2>/dev/null; then AGG35B_RUNNING=true; fi
    if ! $AGG27B_RUNNING && ss -tlnp | grep -q ":8092 " 2>/dev/null; then AGG27B_RUNNING=true; fi
    if ! $Q36_35B_RUNNING && ss -tlnp | grep -q ":8093 " 2>/dev/null; then Q36_35B_RUNNING=true; fi
    if ! $Q36_27B_RUNNING && ss -tlnp | grep -q ":8094 " 2>/dev/null; then Q36_27B_RUNNING=true; fi

    echo "========================================="
    echo "  Stop Qwen3 Servers"
    echo "========================================="
    echo ""
    echo "Running:"
    $CODER_RUNNING      && echo "  ✓ Qwen3-Coder         (port 8085)" || echo "  ✗ Qwen3-Coder         (not running)"
    $VL_RUNNING         && echo "  ✓ Qwen3-VL-32B        (port 8086)" || echo "  ✗ Qwen3-VL-32B        (not running)"
    $B27_RUNNING        && echo "  ✓ Qwen3.5-27B         (port 8087)" || echo "  ✗ Qwen3.5-27B         (not running)"
    $VL4B_RUNNING       && echo "  ✓ Qwen3-VL-4B         (port 8088)" || echo "  ✗ Qwen3-VL-4B         (not running)"
    $B122_RUNNING       && echo "  ✓ Qwen3.5-122B-A10B   (port 8089)" || echo "  ✗ Qwen3.5-122B-A10B   (not running)"
    $OMNICODER_RUNNING  && echo "  ✓ OmniCoder-9B        (port 8090)" || echo "  ✗ OmniCoder-9B        (not running)"
    $AGG35B_RUNNING     && echo "  ✓ Aggressive-35B-A3B  (port 8091)" || echo "  ✗ Aggressive-35B-A3B  (not running)"
    $AGG27B_RUNNING     && echo "  ✓ Aggressive-27B      (port 8092)" || echo "  ✗ Aggressive-27B      (not running)"
    $Q36_35B_RUNNING    && echo "  ✓ Qwen3.6-35B-A3B     (port 8093)" || echo "  ✗ Qwen3.6-35B-A3B     (not running)"
    $Q36_27B_RUNNING    && echo "  ✓ Qwen3.6-27B         (port 8094)" || echo "  ✗ Qwen3.6-27B         (not running)"
    echo ""
    echo " 1) Stop Coder only"
    echo " 2) Stop VL-32B only"
    echo " 3) Stop 27B only"
    echo " 4) Stop VL-4B only"
    echo " 5) Stop 122B only"
    echo " 6) Stop OmniCoder-9B only"
    echo " 7) Stop Aggressive-35B only"
    echo " 8) Stop Aggressive-27B only"
    echo " 9) Stop Qwen3.6-35B-A3B only"
    echo "10) Stop Qwen3.6-27B only"
    echo "11) Stop all"
    echo ""
    read -p "Enter choice [1-11]: " choice
    case "$choice" in
        1)  TARGET="coder" ;;
        2)  TARGET="vl" ;;
        3)  TARGET="27b" ;;
        4)  TARGET="vl4b" ;;
        5)  TARGET="122b" ;;
        6)  TARGET="omnicoder" ;;
        7)  TARGET="agg35b" ;;
        8)  TARGET="agg27b" ;;
        9)  TARGET="q36-35b" ;;
        10) TARGET="q36-27b" ;;
        11) TARGET="all" ;;
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
    vl4b)
        stop_vl4b
        ;;
    27b)
        stop_27b
        ;;
    122b)
        stop_122b
        ;;
    omnicoder)
        stop_omnicoder
        ;;
    agg35b)
        stop_agg35b
        ;;
    agg27b)
        stop_agg27b
        ;;
    q36-35b)
        stop_q36_35b
        ;;
    q36-27b)
        stop_q36_27b
        ;;
    all)
        stop_coder
        stop_vl
        stop_vl4b
        stop_27b
        stop_122b
        stop_omnicoder
        stop_agg35b
        stop_agg27b
        stop_q36_35b
        stop_q36_27b
        kill_all_vllm    # final sweep for any orphaned vLLM workers
        kill_all_sglang  # legacy sweep for any stale SGLang processes
        ;;
    *)
        echo "Usage: $0 [coder|vl|vl4b|27b|122b|omnicoder|agg35b|agg27b|q36-35b|q36-27b|all]"
        exit 1
        ;;
esac

show_gpu_status
echo "Done."