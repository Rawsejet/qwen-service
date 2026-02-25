#!/bin/bash
# Start Qwen3-Coder-Next - Interactive Mode Selection
# Supports both llama.cpp and vLLM backends

LOG_FILE=~/qwen-service/qwen-server.log
PID_FILE=~/qwen-service/qwen-server.pid
BACKEND_FILE=~/qwen-service/qwen-server.backend
PORT=8085

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Server already running with PID $PID"
        exit 1
    fi
fi

echo "========================================="
echo "  Qwen3-Coder-Next Server Configuration"
echo "========================================="
echo ""
echo "Choose inference backend:"
echo ""
echo "1) vLLM  (FP8 weights, better concurrency, tool calling)"
echo "2) llama.cpp  (GGUF weights, single-stream, flexible quant)"
echo ""
read -p "Enter choice [1-2]: " backend_choice

# ─────────────────────────────────────────────
# vLLM PATH
# ─────────────────────────────────────────────
if [ "$backend_choice" == "1" ]; then

    MODEL_PATH=~/models/qwen3/Qwen3-Coder-Next-FP8

    echo ""
    echo "Choose GPU configuration:"
    echo ""
    echo "1) Single GPU"
    echo "   - Max context: ~32K tokens"
    echo "   - Leaves the other GPU free"
    echo ""
    echo "2) Dual GPU - Balanced (131K context)"
    echo "   - Good for responsive interactions"
    echo "   - Uses both GPUs"
    echo ""
    echo "3) Dual GPU - Max Context (256K)"
    echo "   - Maximum context"
    echo "   - High VRAM usage"
    echo ""
    read -p "Enter choice [1-3]: " gpu_choice

    case $gpu_choice in
        1)
            echo ""
            echo "Which GPU to use?"
            echo "  0) GPU 0"
            echo "  1) GPU 1"
            read -p "Enter GPU [0-1]: " gpu_id
            case $gpu_id in
                0|1)
                    CUDA_DEVICES="$gpu_id"
                    OTHER_GPU=$(( 1 - gpu_id ))
                    echo ""
                    echo "Starting on GPU $gpu_id (GPU $OTHER_GPU remains free)..."
                    ;;
                *)
                    echo "Invalid GPU. Exiting."
                    exit 1
                    ;;
            esac
            TP_SIZE=1
            MAX_MODEL_LEN=32768
            GPU_UTIL=0.90
            GPU_LABEL="Single GPU $gpu_id"
            ;;
        2)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MAX_MODEL_LEN=131072
            GPU_UTIL=0.85
            GPU_LABEL="Dual GPU - 131K ctx"
            ;;
        3)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MAX_MODEL_LEN=262144
            GPU_UTIL=0.92
            GPU_LABEL="Dual GPU - 256K ctx"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Configuration:"
    echo "  Backend:     vLLM"
    echo "  Model:       Qwen3-Coder-Next-FP8"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Context:     $MAX_MODEL_LEN tokens"
    echo "  Port:        $PORT"
    echo ""

    # NCCL tuning for RTX PRO 6000 Blackwell (SM120, PCIe PHB topology, no NVLink):
    # P2P/CUMEM and P2P/IPC hang on SM120 - working transport is SHM/direct/direct
    # NCCL_P2P_DISABLE=1 + NCCL_CUMEM_ENABLE=0 forces SHM transport which works
    NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_IB_DISABLE=1 \
    VLLM_ENGINE_CORE_STARTUP_TIMEOUT=300 \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    nohup vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --dtype auto \
        --trust-remote-code \
        --served-model-name "Qwen3-Coder-Next" \
        --disable-custom-all-reduce \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "vllm" > "$BACKEND_FILE"

    echo "Server started with PID $(cat $PID_FILE)"
    echo "Logs: $LOG_FILE"
    echo ""
    echo "Waiting for server to be ready (typically 1-3 min for vLLM)..."
    echo "Press Ctrl+C to background — server will keep running."
    echo ""

    START_TIME=$SECONDS
    LAST_LINE=""
    while true; do
        if ! kill -0 $(cat "$PID_FILE" 2>/dev/null) 2>/dev/null; then
            echo ""
            echo "✗ Server process died. Last 20 lines of log:"
            tail -20 "$LOG_FILE"
            exit 1
        fi
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            ELAPSED=$(( SECONDS - START_TIME ))
            echo ""
            echo "✓ Server is ready! (${ELAPSED}s)"
            echo ""
            echo "Test with:"
            echo "  curl http://127.0.0.1:$PORT/v1/chat/completions \\"
            echo "    -H 'Content-Type: application/json' \\"
            echo "    -d '{\"model\":\"Qwen3-Coder-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
            echo ""
            exit 0
        fi
        NEW_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)
        if [ "$NEW_LINE" != "$LAST_LINE" ]; then
            ELAPSED=$(( SECONDS - START_TIME ))
            echo "  [${ELAPSED}s] $NEW_LINE"
            LAST_LINE="$NEW_LINE"
        fi
        sleep 2
    done

# ─────────────────────────────────────────────
# llama.cpp PATH
# ─────────────────────────────────────────────
elif [ "$backend_choice" == "2" ]; then

    source ~/.cuda13_env

    MODEL_DIR=~/models/qwen3/qwen3-coder-next
    LLAMA_BIN=~/llama.cpp/build/bin/llama-server

    echo ""
    echo "Choose model:"
    echo ""
    echo "1) Q8_0          (Standard 8-bit, faster)"
    echo "2) UD-Q8_K_XL    (Unsloth Dynamic, higher fidelity)"
    echo ""
    read -p "Enter choice [1-2]: " model_choice

    case $model_choice in
        1)
            MODEL_PATH="$MODEL_DIR/Qwen3-Coder-Next-Q8_0/Qwen3-Coder-Next-Q8_0-00001-of-00003.gguf"
            MODEL_NAME="Q8_0"
            ;;
        2)
            MODEL_PATH="$MODEL_DIR/Qwen3-Coder-Next-UD-Q8_K_XL/Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003.gguf"
            MODEL_NAME="UD-Q8_K_XL"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model file not found: $MODEL_PATH"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo ""
    echo "1) Single GPU"
    echo "   - All model + KV cache on one GPU"
    echo "   - Max context: ~170K tokens"
    echo "   - Leaves the other GPU free"
    echo ""
    echo "2) Dual GPU (Split across both)"
    echo "   - Max context: ~256K tokens"
    echo "   - Uses both GPUs"
    echo ""
    read -p "Enter choice [1-2]: " gpu_choice

    echo ""
    echo "Choose context size:"
    echo ""
    echo "1) Balanced (98K single / 131K dual)"
    echo "2) Maximum  (174K single / 256K dual)"
    echo ""
    read -p "Enter choice [1-2]: " ctx_choice

    case $gpu_choice in
        1)
            echo ""
            echo "Which GPU to use?"
            echo "  0) GPU 0"
            echo "  1) GPU 1"
            read -p "Enter GPU [0-1]: " gpu_id
            case $gpu_id in
                0|1)
                    CUDA_DEVICES="$gpu_id"
                    OTHER_GPU=$(( 1 - gpu_id ))
                    echo ""
                    echo "Starting on GPU $gpu_id (GPU $OTHER_GPU remains free)..."
                    ;;
                *)
                    echo "Invalid GPU. Exiting."
                    exit 1
                    ;;
            esac
            GPU_LAYERS=999
            TENSOR_SPLIT=""
            SPLIT_LABEL="Single GPU $gpu_id"
            case $ctx_choice in
                1) CTX_SIZE=98304  ;;
                2) CTX_SIZE=174762 ;;
                *) CTX_SIZE=98304  ;;
            esac
            ;;
        2)
            echo ""
            echo "Choose tensor-split ratio:"
            echo "  1) 50/50   2) 60/40   3) 65/35   4) 70/30   5) 75/25"
            read -p "Enter choice [1-5]: " split_choice
            case $split_choice in
                1) TENSOR_SPLIT="--tensor-split 50,50"; SPLIT_LABEL="50/50" ;;
                2) TENSOR_SPLIT="--tensor-split 60,40"; SPLIT_LABEL="60/40" ;;
                3) TENSOR_SPLIT="--tensor-split 65,35"; SPLIT_LABEL="65/35" ;;
                4) TENSOR_SPLIT="--tensor-split 70,30"; SPLIT_LABEL="70/30" ;;
                5) TENSOR_SPLIT="--tensor-split 75,25"; SPLIT_LABEL="75/25" ;;
                *) TENSOR_SPLIT="--tensor-split 70,30"; SPLIT_LABEL="70/30" ;;
            esac
            GPU_LAYERS=999
            CUDA_DEVICES="0,1"
            case $ctx_choice in
                1) CTX_SIZE=131072 ;;
                2) CTX_SIZE=262144 ;;
                *) CTX_SIZE=131072 ;;
            esac
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Configuration:"
    echo "  Backend:     llama.cpp"
    echo "  Model:       $MODEL_NAME"
    echo "  GPUs:        $SPLIT_LABEL"
    echo "  Context:     $CTX_SIZE tokens"
    echo "  Port:        $PORT"
    echo ""

    # Auto-expand swap for large single-GPU configs
    if [ -z "$TENSOR_SPLIT" ] && ([ "$CTX_SIZE" -gt 170000 ] || [[ "$MODEL_PATH" == *"UD-Q8_K_XL"* ]]); then
        CURRENT_SWAP=$(grep "^/swap.img" /proc/swaps | awk '{print $3}')
        CURRENT_SWAP=${CURRENT_SWAP:-0}
        CURRENT_SWAP_GB=$((CURRENT_SWAP / 1048576))
        if [ "$CURRENT_SWAP_GB" -lt 16 ]; then
            echo "⚠️  Expanding swap to 16GB for this configuration..."
            sudo fallocate -l 16G /swap_new.img 2>/dev/null
            sudo chmod 600 /swap_new.img 2>/dev/null
            sudo mkswap /swap_new.img 2>/dev/null
            sudo swapon /swap_new.img 2>/dev/null
            sudo swapoff /swap.img 2>/dev/null
            sudo rm /swap.img 2>/dev/null
            sudo mv /swap_new.img /swap.img 2>/dev/null
            echo "✓ Swap expanded to 16GB"
            echo ""
        fi
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup "$LLAMA_BIN" \
        --model "$MODEL_PATH" \
        --jinja \
        --n-gpu-layers $GPU_LAYERS \
        $TENSOR_SPLIT \
        --ctx-size $CTX_SIZE \
        --flash-attn on \
        --cache-type-k f16 \
        --cache-type-v f16 \
        --temp 1.0 \
        --top-p 0.95 \
        --top-k 40 \
        --min_p 0.01 \
        --host 127.0.0.1 \
        --port $PORT \
        --threads 16 \
        --batch-size 4096 \
        --ubatch-size 1024 \
        --poll 100 \
        --mlock \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "llama.cpp" > "$BACKEND_FILE"

    echo "Server started with PID $(cat $PID_FILE)"
    echo "Logs: $LOG_FILE"
    echo ""
    echo "Waiting for server to be ready..."

    for i in {1..60}; do
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            echo ""
            echo "✓ Server is ready!"
            echo ""
            echo "Test with:"
            echo "  curl http://127.0.0.1:$PORT/v1/chat/completions \\"
            echo "    -H 'Content-Type: application/json' \\"
            echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
            echo ""
            exit 0
        fi
        sleep 1
    done
    echo "Server may still be starting. Check logs: ./logs-qwen.sh"

else
    echo "Invalid choice. Exiting."
    exit 1
fi