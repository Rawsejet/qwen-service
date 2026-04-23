#!/bin/bash
# Start Qwen3 Model Server - Interactive Mode Selection
# Supports: Qwen3-Coder-Next-FP8 (port 8085), Qwen3-VL-32B-FP8 (port 8086)
# Backends: vLLM, llama.cpp

LOG_DIR=~/qwen-service

# Source shared model registry
source "$LOG_DIR/models.conf"

# Derive convenience aliases from the registry
PORT_CODER=${MODEL_PORTS[1]}
PORT_VL4B=${MODEL_PORTS[2]}
PORT_VL=${MODEL_PORTS[3]}
PORT_27B=${MODEL_PORTS[4]}
PORT_122B=${MODEL_PORTS[5]}
PID_FILE_CODER=$LOG_DIR/${MODEL_PIDS[1]}
PID_FILE_VL4B=$LOG_DIR/${MODEL_PIDS[2]}
PID_FILE_VL=$LOG_DIR/${MODEL_PIDS[3]}
PID_FILE_27B=$LOG_DIR/${MODEL_PIDS[4]}
PID_FILE_122B=$LOG_DIR/${MODEL_PIDS[5]}
PORT_OMNICODER=${MODEL_PORTS[6]}
PID_FILE_OMNICODER=$LOG_DIR/${MODEL_PIDS[6]}
PORT_AGG35B=${MODEL_PORTS[7]}
PID_FILE_AGG35B=$LOG_DIR/${MODEL_PIDS[7]}
PORT_AGG27B=${MODEL_PORTS[8]}
PID_FILE_AGG27B=$LOG_DIR/${MODEL_PIDS[8]}
PORT_Q36_35B=${MODEL_PORTS[9]}
PID_FILE_Q36_35B=$LOG_DIR/${MODEL_PIDS[9]}
PORT_Q36_27B=${MODEL_PORTS[10]}
PID_FILE_Q36_27B=$LOG_DIR/${MODEL_PIDS[10]}
BACKEND_FILE=$LOG_DIR/qwen-server.backend

# NCCL tuning for RTX PRO 6000 Blackwell (SM120, PCIe PHB topology, no NVLink):
# P2P/CUMEM and P2P/IPC hang on SM120 - working transport is SHM/direct/direct
# NCCL_P2P_DISABLE=1 + NCCL_CUMEM_ENABLE=0 forces SHM transport which works
# --disable-custom-all-reduce required: vLLM's custom all-reduce IPC also hangs on SM120
NCCL_ENV="NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_IB_DISABLE=1 VLLM_ENGINE_CORE_STARTUP_TIMEOUT=300 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

wait_for_vllm() {
    local PORT=$1
    local PID_FILE=$2
    local LOG_FILE=$3
    local START_TIME=$SECONDS
    local LAST_LINE=""
    echo "Waiting for server to be ready (typically 1-3 min for vLLM)..."
    echo "Press Ctrl+C to background — server will keep running."
    echo ""
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
            echo "  curl http://127.0.0.1:$PORT/v1/models | python3 -m json.tool"
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
}

# Like wait_for_vllm but returns (instead of exit 0) so caller can run post-ready steps (e.g. warmup)
wait_for_vllm_then_continue() {
    local PORT=$1
    local PID_FILE=$2
    local LOG_FILE=$3
    local START_TIME=$SECONDS
    local LAST_LINE=""
    echo "Waiting for server to be ready (typically 1-3 min for vLLM)..."
    echo "Press Ctrl+C to background — server will keep running."
    echo ""
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
            return 0
        fi
        NEW_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)
        if [ "$NEW_LINE" != "$LAST_LINE" ]; then
            ELAPSED=$(( SECONDS - START_TIME ))
            echo "  [${ELAPSED}s] $NEW_LINE"
            LAST_LINE="$NEW_LINE"
        fi
        sleep 2
    done
}

echo "========================================="
echo "  Qwen3 Model Server"
echo "========================================="
echo ""
echo "Choose model:"
echo ""
for i in $(seq 1 $MODEL_COUNT); do
    printf "%d) %-22s (port %s)\n" "$i" "${MODEL_NAMES[$i]}" "${MODEL_PORTS[$i]}"
    echo "   ${MODEL_DESCS[$i]}"
    echo ""
done
read -p "Enter choice [1-$MODEL_COUNT]: " model_choice

# ─────────────────────────────────────────────
# QWEN3-CODER-NEXT
# ─────────────────────────────────────────────
if [ "$model_choice" == "1" ]; then

    if [ -f "$PID_FILE_CODER" ]; then
        PID=$(cat "$PID_FILE_CODER")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Coder server already running with PID $PID"
            exit 1
        fi
    fi

    echo ""
    echo "Choose inference backend:"
    echo ""
    echo "1) vLLM  (FP8 weights, better concurrency, tool calling)"
    echo "2) llama.cpp  (GGUF weights, single-stream, flexible quant)"
    echo ""
    read -p "Enter choice [1-2]: " backend_choice

    # ── vLLM ──────────────────────────────────
    if [ "$backend_choice" == "1" ]; then

        MODEL_PATH=~/models/qwen3/Qwen3-Coder-Next-FP8
        LOG_FILE=$LOG_DIR/qwen-coder-vllm.log

        echo ""
        echo "Choose GPU configuration:"
        echo ""
        echo "1) Single GPU"
        echo "   - Max context: ~32K tokens"
        echo "   - Leaves the other GPU free"
        echo ""
        echo "2) Dual GPU - Balanced (131K context, 0.85 util)"
        echo "   - Full KV cache, good for interactive use"
        echo "   - Uses both GPUs at 85% VRAM"
        echo ""
        echo "3) Dual GPU - Shared mode (131K context, 0.60 util)"
        echo "   - Leaves ~38GB free per GPU for VL model"
        echo "   - Run alongside Qwen3-VL-32B on same GPUs"
        echo ""
        echo "4) Dual GPU - Max Context (256K)"
        echo "   - Maximum context length"
        echo "   - High VRAM usage"
        echo ""
        read -p "Enter choice [1-4]: " gpu_choice

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
                GPU_LABEL="Dual GPU - 131K ctx (0.85 util)"
                ;;
            3)
                CUDA_DEVICES="0,1"
                TP_SIZE=2
                MAX_MODEL_LEN=131072
                GPU_UTIL=0.60
                GPU_LABEL="Dual GPU - Shared mode (0.60 util, ~38GB free per GPU)"
                echo ""
                echo "Note: Start Qwen3-VL-32B after this with option 2 of this script."
                ;;
            4)
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
        echo "  Port:        $PORT_CODER"
        echo ""

        env $NCCL_ENV \
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
        nohup vllm serve "$MODEL_PATH" \
            --host 127.0.0.1 \
            --port $PORT_CODER \
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

        echo $! > "$PID_FILE_CODER"
        echo "vllm" > "$BACKEND_FILE"
        echo "Server started with PID $(cat $PID_FILE_CODER)"
        echo "Logs: $LOG_FILE"
        echo ""
        wait_for_vllm $PORT_CODER $PID_FILE_CODER $LOG_FILE

    # ── llama.cpp ─────────────────────────────
    elif [ "$backend_choice" == "2" ]; then

        source ~/.cuda13_env

        MODEL_DIR=~/models/qwen3/qwen3-coder-next
        LLAMA_BIN=~/llama.cpp/build/bin/llama-server
        LOG_FILE=$LOG_DIR/qwen-coder-llama.log

        echo ""
        echo "Choose model:"
        echo ""
        echo "1) Q8_0          (Standard 8-bit, faster)"
        echo "2) UD-Q8_K_XL    (Unsloth Dynamic, higher fidelity)"
        echo ""
        read -p "Enter choice [1-2]: " model_choice_llama

        case $model_choice_llama in
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
        echo "   - Max context: ~170K tokens"
        echo "   - Leaves the other GPU free"
        echo ""
        echo "2) Dual GPU"
        echo "   - Max context: ~256K tokens"
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
        echo "Choose concurrent requests (parallel slots):"
        echo ""
        echo "1) 1 slot  - Single request at a time (lowest memory)"
        echo "2) 2 slots - 2 concurrent requests (~2x context memory)"
        echo "3) 4 slots - 4 concurrent requests (~4x context memory)"
        echo "4) 8 slots - 8 concurrent requests (~8x context memory)"
        echo ""
        read -p "Enter choice [1-4]: " parallel_choice

        case $parallel_choice in
            1) PARALLEL_SLOTS=1 ;;
            2) PARALLEL_SLOTS=2 ;;
            3) PARALLEL_SLOTS=4 ;;
            4) PARALLEL_SLOTS=8 ;;
            *)
                echo "Invalid choice. Using 2 slots."
                PARALLEL_SLOTS=2
                ;;
        esac

        echo ""
        echo "Configuration:"
        echo "  Backend:     llama.cpp"
        echo "  Model:       $MODEL_NAME"
        echo "  GPUs:        $SPLIT_LABEL"
        echo "  Context:     $CTX_SIZE tokens"
        echo "  Parallel:    $PARALLEL_SLOTS slot(s)"
        echo "  Port:        $PORT_CODER"
        echo ""

        if [ "$PARALLEL_SLOTS" -ge 4 ] && [ "$CTX_SIZE" -ge 131072 ]; then
            echo "  WARNING: High context + 4+ parallel slots = very high KV cache memory"
            echo ""
        fi

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
            --parallel $PARALLEL_SLOTS \
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
            --port $PORT_CODER \
            --threads 16 \
            --batch-size 4096 \
            --ubatch-size 1024 \
            --poll 100 \
            --mlock \
            > "$LOG_FILE" 2>&1 &

        echo $! > "$PID_FILE_CODER"
        echo "llama.cpp" > "$BACKEND_FILE"
        echo "Server started with PID $(cat $PID_FILE_CODER)"
        echo "Logs: $LOG_FILE"
        echo ""
        echo "Waiting for server to be ready..."
        for i in {1..60}; do
            if curl -s http://127.0.0.1:$PORT_CODER/health > /dev/null 2>&1; then
                echo "✓ Server is ready!"
                exit 0
            fi
            sleep 1
        done
        echo "Server may still be starting. Check logs: tail -f $LOG_FILE"

    else
        echo "Invalid choice. Exiting."
        exit 1
    fi

# ─────────────────────────────────────────────
# QWEN3-VL-4B
# ─────────────────────────────────────────────
elif [ "$model_choice" == "2" ]; then

    if [ -f "$PID_FILE_VL4B" ]; then
        PID=$(cat "$PID_FILE_VL4B")
        if ps -p $PID > /dev/null 2>&1; then
            echo "VL-4B server already running with PID $PID"
            exit 1
        fi
    fi

    MODEL_PATH=~/models/qwen3/Qwen3-VL-4B-Instruct-FP8
    LOG_FILE=$LOG_DIR/qwen-vl4b-vllm.log

    if [ ! -d "$MODEL_PATH" ]; then
        echo ""
        echo "Model not found at $MODEL_PATH"
        echo "Download with:"
        echo "  huggingface-cli download Qwen/Qwen3-VL-4B-Instruct-FP8 --local-dir $MODEL_PATH"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo ""
    echo "1) Single GPU (0.85 util, ~131K context)"
    echo "   - Run alongside Coder on the other GPU"
    echo ""
    echo "2) Single GPU (0.30 util, ~64K context)"
    echo "   - Lower util, more headroom on the GPU"
    echo ""
    echo "3) Dual GPU - Solo (0.85 util, ~262K context)"
    echo "   - Max context, both GPUs"
    echo ""
    echo "4) Dual GPU - Minimal (0.03 util, ~8K context)"
    echo "   - Tiny footprint, runs alongside heavy models on both GPUs"
    echo ""
    read -p "Enter choice [1-4]: " gpu_choice

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
            GPU_UTIL=0.85
            MAX_MODEL_LEN=131072
            GPU_LABEL="Single GPU $gpu_id (0.85 util)"
            ;;
        2)
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
            GPU_UTIL=0.30
            MAX_MODEL_LEN=32768
            GPU_LABEL="Single GPU $gpu_id (0.30 util)"
            ;;
        3)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            GPU_UTIL=0.85
            MAX_MODEL_LEN=262144
            GPU_LABEL="Dual GPU - Solo (0.85 util)"
            ;;
        4)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            GPU_UTIL=0.03
            MAX_MODEL_LEN=8192
            GPU_LABEL="Dual GPU - Minimal (0.03 util)"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Configuration:"
    echo "  Backend:     vLLM"
    echo "  Model:       Qwen3-VL-4B-FP8"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Context:     $MAX_MODEL_LEN tokens"
    echo "  Port:        $PORT_VL4B"
    echo ""

    env $NCCL_ENV \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    nohup vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port $PORT_VL4B \
        --tensor-parallel-size $TP_SIZE \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --dtype auto \
        --trust-remote-code \
        --served-model-name "Qwen3-VL-4B" \
        --disable-custom-all-reduce \
        --allowed-local-media-path / \
        --limit-mm-per-prompt '{"image": 5, "video": 1}' \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE_VL4B"
    echo "Server started with PID $(cat $PID_FILE_VL4B)"
    echo "Logs: $LOG_FILE"
    echo ""
    wait_for_vllm $PORT_VL4B $PID_FILE_VL4B $LOG_FILE

# ─────────────────────────────────────────────
# QWEN3-VL-32B
# ─────────────────────────────────────────────
elif [ "$model_choice" == "3" ]; then

    if [ -f "$PID_FILE_VL" ]; then
        PID=$(cat "$PID_FILE_VL")
        if ps -p $PID > /dev/null 2>&1; then
            echo "VL server already running with PID $PID"
            exit 1
        fi
    fi

    MODEL_PATH=~/models/qwen3/Qwen3-VL-32B-Instruct-FP8
    LOG_FILE=$LOG_DIR/qwen-vl-vllm.log

    if [ ! -d "$MODEL_PATH" ]; then
        echo ""
        echo "Model not found at $MODEL_PATH"
        echo "Download with:"
        echo "  huggingface-cli download Qwen/Qwen3-VL-32B-Thinking-FP8 --local-dir $MODEL_PATH"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo ""
    echo "1) Dual GPU - Solo (0.85 util, ~64K context)"
    echo "   - Full GPU memory for VL model"
    echo "   - Coder model should NOT be running"
    echo ""
    echo "2) Dual GPU - Shared mode (0.55 util, ~32K context)"
    echo "   - Leaves room for Coder in shared mode (0.60 util)"
    echo "   - Start Coder first with option 1 → vLLM → Shared mode"
    echo ""
    read -p "Enter choice [1-2]: " gpu_choice

    case $gpu_choice in
        1)
            GPU_UTIL=0.85
            MAX_MODEL_LEN=65536
            GPU_LABEL="Dual GPU - Solo (0.85 util)"
            ;;
        2)
            GPU_UTIL=0.55
            MAX_MODEL_LEN=32768
            GPU_LABEL="Dual GPU - Shared mode (0.55 util)"
            echo ""
            echo "Note: Make sure Coder is running in shared mode (0.60 util) on port $PORT_CODER"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Configuration:"
    echo "  Backend:     vLLM"
    echo "  Model:       Qwen3-VL-32B-FP8"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Context:     $MAX_MODEL_LEN tokens"
    echo "  Port:        $PORT_VL"
    echo ""

    env $NCCL_ENV \
    CUDA_VISIBLE_DEVICES=0,1 \
    nohup vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port $PORT_VL \
        --tensor-parallel-size 2 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --dtype auto \
        --trust-remote-code \
        --served-model-name "Qwen3-VL-32B" \
        --disable-custom-all-reduce \
        --limit-mm-per-prompt '{"image": 5, "video": 1}' \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE_VL"
    echo "Server started with PID $(cat $PID_FILE_VL)"
    echo "Logs: $LOG_FILE"
    echo ""
    wait_for_vllm $PORT_VL $PID_FILE_VL $LOG_FILE

elif [ "$model_choice" == "4" ]; then

    if [ -f "$PID_FILE_27B" ]; then
        PID=$(cat "$PID_FILE_27B")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Qwen3.5-27B server already running with PID $PID"
            exit 1
        fi
    fi

    echo ""
    echo "Choose model variant:"
    echo ""
    echo "1) Qwen3.5-27B-FP8    (~27GB, more context headroom)"
    echo "2) Qwen3.5-27B        (~54GB BF16, higher fidelity)"
    echo ""
    read -p "Enter choice [1-2]: " variant_choice

    case $variant_choice in
        1)
            MODEL_PATH=~/models/qwen3/Qwen3.5-27B-FP8
            MODEL_LABEL="Qwen3.5-27B-FP8"
            DOWNLOAD_ID="Qwen/Qwen3.5-27B-FP8"
            ;;
        2)
            MODEL_PATH=~/models/qwen3/Qwen3.5-27B
            MODEL_LABEL="Qwen3.5-27B (BF16)"
            DOWNLOAD_ID="Qwen/Qwen3.5-27B"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    LOG_FILE=$LOG_DIR/qwen-27b-vllm.log

    if [ ! -d "$MODEL_PATH" ]; then
        echo ""
        echo "Model not found at $MODEL_PATH"
        echo "Download with:"
        echo "  huggingface-cli download $DOWNLOAD_ID --local-dir $MODEL_PATH"
        exit 1
    fi

    if [ "$variant_choice" == "1" ]; then
        # FP8 variant — same as before
        echo ""
        echo "Choose GPU configuration:"
        echo ""
        echo "1) Single GPU (0.80 mem, ~131K context)"
        echo "   - Leaves other GPU free"
        echo ""
        echo "2) Dual GPU - Solo (0.80 mem, ~262K context)"
        echo "   - Split across both GPUs for max context"
        echo "   - Coder model should NOT be running"
        echo ""
        echo "3) Dual GPU - Shared mode (0.55 mem, ~131K context)"
        echo "   - Leaves room for Coder in shared mode (0.60 util)"
        echo "   - Start Coder first with option 1 → vLLM → Shared mode"
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
                        ;;
                    *)
                        echo "Invalid GPU. Exiting."
                        exit 1
                        ;;
                esac
                TP_SIZE=1
                MEM_FRAC=0.80
                MAX_MODEL_LEN=131072
                GPU_LABEL="Single GPU $gpu_id"
                ;;
            2)
                CUDA_DEVICES="0,1"
                TP_SIZE=2
                MEM_FRAC=0.80
                MAX_MODEL_LEN=262144
                GPU_LABEL="Dual GPU - Solo"
                ;;
            3)
                CUDA_DEVICES="0,1"
                TP_SIZE=2
                MEM_FRAC=0.55
                MAX_MODEL_LEN=131072
                GPU_LABEL="Dual GPU - Shared mode"
                echo ""
                echo "Note: Make sure Coder is running in shared mode (0.60 util) on port $PORT_CODER"
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac

    else
        # BF16 variant — reduced context due to ~2x model size
        echo ""
        echo "Choose GPU configuration:"
        echo ""
        echo "1) Single GPU (0.85 mem, ~65K context)"
        echo "   - Leaves other GPU free"
        echo ""
        echo "2) Dual GPU - Solo (0.85 mem, ~141K context)"
        echo "   - Split across both GPUs for max context"
        echo "   - Coder model should NOT be running"
        echo ""
        echo "3) Dual GPU - Shared mode (0.55 mem, ~65K context)"
        echo "   - Leaves room for Coder in shared mode (0.60 util)"
        echo "   - Start Coder first with option 1 → vLLM → Shared mode"
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
                        ;;
                    *)
                        echo "Invalid GPU. Exiting."
                        exit 1
                        ;;
                esac
                TP_SIZE=1
                MEM_FRAC=0.85
                MAX_MODEL_LEN=65536
                GPU_LABEL="Single GPU $gpu_id"
                ;;
            2)
                CUDA_DEVICES="0,1"
                TP_SIZE=2
                MEM_FRAC=0.85
                MAX_MODEL_LEN=131072
                GPU_LABEL="Dual GPU - Solo"
                ;;
            3)
                CUDA_DEVICES="0,1"
                TP_SIZE=2
                MEM_FRAC=0.55
                MAX_MODEL_LEN=65536
                GPU_LABEL="Dual GPU - Shared mode"
                echo ""
                echo "Note: Make sure Coder is running in shared mode (0.60 util) on port $PORT_CODER"
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
    fi

    echo ""
    echo "Choose mode:"
    echo ""
    echo "1) Tools             tool calling + reasoning  (use with Claude Code)"
    echo "2) Tools + MTP       tools + speculative decoding (faster, ~2x tokens/s)"
    echo "3) Text only         reasoning only, no tool calling, no vision"
    echo "                     (skips vision encoder, frees VRAM for KV cache)"
    echo ""
    read -p "Enter choice [1-3]: " mode_choice

    case $mode_choice in
        1)
            MODE_FLAGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
            MODE_SPECULATIVE=""
            MODE_LABEL="Tools"
            ;;
        2)
            MODE_FLAGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
            echo ""
            echo "Choose MTP tokens:"
            echo "  1) 3 tokens (multi-user)"
            echo "  2) 5 tokens (single-user)"
            read -p "Enter choice [1-2]: " mtp_choice
            case $mtp_choice in
                2) MODE_SPECULATIVE='{"method":"mtp","num_speculative_tokens":5}' ;;
                *) MODE_SPECULATIVE='{"method":"mtp","num_speculative_tokens":3}' ;;
            esac
            MODE_LABEL="Tools + MTP"
            ;;
        3)
            MODE_FLAGS=(--reasoning-parser qwen3 --language-model-only)
            MODE_SPECULATIVE=""
            MODE_LABEL="Text only"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Choose sampling preset:"
    echo ""
    echo "1) Thinking / General   temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo "2) Thinking / Coding    temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,  repetition_penalty=1.0"
    echo "3) Instruct / General   temp=0.7, top_p=0.8,  top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo "4) Instruct / Reasoning temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo ""
    read -p "Enter choice [1-4]: " sampling_choice

    case $sampling_choice in
        1)
            SAMPLING_LABEL="Thinking/General"
            SAMPLING_PARAMS="temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=true
            ;;
        2)
            SAMPLING_LABEL="Thinking/Coding"
            SAMPLING_PARAMS="temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0"
            ENABLE_THINKING=true
            ;;
        3)
            SAMPLING_LABEL="Instruct/General"
            SAMPLING_PARAMS="temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=false
            ;;
        4)
            SAMPLING_LABEL="Instruct/Reasoning"
            SAMPLING_PARAMS="temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=false
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    # Save preset for reference
    cat > ~/qwen-service/qwen-27b-sampling.conf << CONF
SAMPLING_LABEL=$SAMPLING_LABEL
SAMPLING_PARAMS=$SAMPLING_PARAMS
ENABLE_THINKING=$ENABLE_THINKING
CONF

    echo ""
    echo "Configuration:"
    echo "  Backend:     vLLM  (conda env: vllm)"
    echo "  Model:       $MODEL_LABEL"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Mode:        $MODE_LABEL"
    echo "  Sampling:    $SAMPLING_LABEL  ($SAMPLING_PARAMS)"
    echo "  Context:     $MAX_MODEL_LEN tokens"
    echo "  Port:        $PORT_27B"
    echo ""
    echo "  Note: /think and /no_think are NOT supported on Qwen3.5."
    echo "        Use enable_thinking=$ENABLE_THINKING in chat_template_kwargs."
    echo ""

    # Build speculative config arg as array to preserve JSON quoting
    SPEC_ARGS=()
    if [ -n "$MODE_SPECULATIVE" ]; then
        SPEC_ARGS=("--speculative-config" "$MODE_SPECULATIVE")
    fi

    # Activate env directly instead of conda run to preserve argument quoting
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate vllm

    env $NCCL_ENV \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    nohup vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port $PORT_27B \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $MEM_FRAC \
        --max-model-len $MAX_MODEL_LEN \
        --served-model-name "Qwen3.5-27B" \
        --dtype auto \
        --trust-remote-code \
        --attention-backend FLASH_ATTN \
        --disable-custom-all-reduce \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        "${MODE_FLAGS[@]}" \
        "${SPEC_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE_27B"
    echo "vllm" > ~/qwen-service/qwen-27b-server.backend
    echo "Server started with PID $(cat $PID_FILE_27B)"
    echo "Logs: $LOG_FILE"
    echo ""
    wait_for_vllm_then_continue $PORT_27B $PID_FILE_27B $LOG_FILE

    # Warmup: pay the ~50s Triton compilation cost now, not on the first real request
    echo "Running warmup request (Triton kernel compilation, ~50s)..."
    WARMUP_START=$SECONDS
    curl -s http://127.0.0.1:$PORT_27B/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"Qwen3.5-27B\",
            \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
            \"max_tokens\": 5,
            \"chat_template_kwargs\": {\"enable_thinking\": true}
        }" > /dev/null 2>&1
    WARMUP_ELAPSED=$(( SECONDS - WARMUP_START ))
    echo "✓ Warmup done (${WARMUP_ELAPSED}s). Server is ready for real requests."
    echo ""
    echo "Test with:"
    echo "  curl http://127.0.0.1:$PORT_27B/v1/models | python3 -m json.tool"
    echo ""

elif [ "$model_choice" == "5" ]; then

# ─────────────────────────────────────────────
# QWEN3.5-122B-A10B
# ─────────────────────────────────────────────

    if [ -f "$PID_FILE_122B" ]; then
        PID=$(cat "$PID_FILE_122B")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Qwen3.5-122B-A10B server already running with PID $PID"
            exit 1
        fi
    fi

    MODEL_PATH=~/models/qwen3/Qwen3.5-122B-A10B-FP8
    LOG_FILE=$LOG_DIR/qwen-122b-vllm.log

    if [ ! -d "$MODEL_PATH" ]; then
        echo ""
        echo "Model not found at $MODEL_PATH"
        echo "Download with:"
        echo "  huggingface-cli download Qwen/Qwen3.5-122B-A10B-FP8 --local-dir $MODEL_PATH"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo ""
    echo "1) Dual GPU - Solo Max (0.90 util, ~256K context)"
    echo "   - Maximum context, high VRAM usage"
    echo "   - Coder model must NOT be running"
    echo ""
    echo "2) Dual GPU - Solo High (0.90 util, ~128K context)"
    echo "   - High context with slightly lower VRAM pressure"
    echo "   - Coder model must NOT be running"
    echo ""
    echo "3) Dual GPU - Solo Medium (0.75 util, ~64K context)"
    echo "   - Balanced: good context, stable performance"
    echo "   - Coder model must NOT be running"
    echo ""
    echo "4) Dual GPU - Shared mode (0.65 util, ~32K context)"
    echo "   - Leaves room for Coder in shared mode (0.60 util)"
    echo "   - Start Coder first with option 1 → vLLM → Shared mode"
    echo ""
    read -p "Enter choice [1-4]: " gpu_choice

    case $gpu_choice in
        1)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.90
            MAX_MODEL_LEN=262144
            GPU_LABEL="Dual GPU - Solo Max (0.90 util)"
            ;;
        2)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.90
            MAX_MODEL_LEN=131072
            GPU_LABEL="Dual GPU - Solo High (0.90 util)"
            ;;
        3)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.75
            MAX_MODEL_LEN=65536
            GPU_LABEL="Dual GPU - Solo Medium (0.75 util)"
            ;;
        4)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.65
            MAX_MODEL_LEN=32768
            GPU_LABEL="Dual GPU - Shared mode (0.65 util)"
            echo ""
            echo "Note: Make sure Coder is running in shared mode (0.60 util) on port $PORT_CODER"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Choose mode:"
    echo ""
    echo "1) Tools             tool calling + reasoning  (use with Claude Code)"
    echo "2) Tools + MTP       tools + speculative decoding (faster, ~2x tokens/s)"
    echo "3) Text only         reasoning only, no tool calling"
    echo ""
    read -p "Enter choice [1-3]: " mode_choice

    case $mode_choice in
        1)
            MODE_FLAGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
            MODE_SPECULATIVE=""
            MODE_LABEL="Tools"
            ;;
        2)
            MODE_FLAGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
            echo ""
            echo "Choose MTP tokens:"
            echo "  1) 3 tokens (multi-user)"
            echo "  2) 5 tokens (single-user)"
            read -p "Enter choice [1-2]: " mtp_choice
            case $mtp_choice in
                2) MODE_SPECULATIVE='{"method":"mtp","num_speculative_tokens":5}' ;;
                *) MODE_SPECULATIVE='{"method":"mtp","num_speculative_tokens":3}' ;;
            esac
            MODE_LABEL="Tools + MTP"
            ;;
        3)
            MODE_FLAGS=(--reasoning-parser qwen3)
            MODE_SPECULATIVE=""
            MODE_LABEL="Text only"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Choose sampling preset:"
    echo ""
    echo "1) Thinking / General   temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo "2) Thinking / Coding    temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,  repetition_penalty=1.0"
    echo "3) Instruct / General   temp=0.7, top_p=0.8,  top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo "4) Instruct / Reasoning temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo ""
    read -p "Enter choice [1-4]: " sampling_choice

    case $sampling_choice in
        1)
            SAMPLING_LABEL="Thinking/General"
            SAMPLING_PARAMS="temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=true
            ;;
        2)
            SAMPLING_LABEL="Thinking/Coding"
            SAMPLING_PARAMS="temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0"
            ENABLE_THINKING=true
            ;;
        3)
            SAMPLING_LABEL="Instruct/General"
            SAMPLING_PARAMS="temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=false
            ;;
        4)
            SAMPLING_LABEL="Instruct/Reasoning"
            SAMPLING_PARAMS="temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=false
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Configuration:"
    echo "  Backend:     vLLM  (conda env: vllm)"
    echo "  Model:       Qwen3.5-122B-A10B-FP8"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Mode:        $MODE_LABEL"
    echo "  Sampling:    $SAMPLING_LABEL  ($SAMPLING_PARAMS)"
    echo "  Context:     $MAX_MODEL_LEN tokens"
    echo "  Port:        $PORT_122B"
    echo ""
    echo "  Note: /think and /no_think are NOT supported on Qwen3.5."
    echo "        Use enable_thinking=$ENABLE_THINKING in chat_template_kwargs."
    echo ""

    SPEC_ARGS=()
    if [ -n "$MODE_SPECULATIVE" ]; then
        SPEC_ARGS=("--speculative-config" "$MODE_SPECULATIVE")
    fi

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate vllm

    env $NCCL_ENV \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    nohup vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port $PORT_122B \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $MEM_FRAC \
        --max-model-len $MAX_MODEL_LEN \
        --served-model-name "Qwen3.5-122B-A10B" \
        --dtype auto \
        --trust-remote-code \
        --attention-backend FLASH_ATTN \
        --disable-custom-all-reduce \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        "${MODE_FLAGS[@]}" \
        "${SPEC_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE_122B"
    echo "vllm" > ~/qwen-service/qwen-122b-server.backend
    echo "Server started with PID $(cat $PID_FILE_122B)"
    echo "Logs: $LOG_FILE"
    echo ""
    wait_for_vllm_then_continue $PORT_122B $PID_FILE_122B $LOG_FILE

    echo "Running warmup request (Triton kernel compilation, ~50s)..."
    WARMUP_START=$SECONDS
    curl -s http://127.0.0.1:$PORT_122B/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"Qwen3.5-122B-A10B\",
            \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
            \"max_tokens\": 5,
            \"chat_template_kwargs\": {\"enable_thinking\": true}
        }" > /dev/null 2>&1
    WARMUP_ELAPSED=$(( SECONDS - WARMUP_START ))
    echo "✓ Warmup done (${WARMUP_ELAPSED}s). Server is ready for real requests."
    echo ""
    echo "Test with:"
    echo "  curl http://127.0.0.1:$PORT_122B/v1/models | python3 -m json.tool"
    echo ""

elif [ "$model_choice" == "6" ]; then

# ─────────────────────────────────────────────
# OMNICODER-9B
# ─────────────────────────────────────────────

    if [ -f "$PID_FILE_OMNICODER" ]; then
        PID=$(cat "$PID_FILE_OMNICODER")
        if ps -p $PID > /dev/null 2>&1; then
            echo "OmniCoder-9B server already running with PID $PID"
            exit 1
        fi
    fi

    MODEL_PATH=~/models/qwen3/omnicoder-9b
    LOG_FILE=$LOG_DIR/omnicoder-9b-vllm.log

    if [ ! -d "$MODEL_PATH" ]; then
        echo ""
        echo "Model not found at $MODEL_PATH"
        echo "Download with:"
        echo "  huggingface-cli download Tesslate/OmniCoder-9B --local-dir $MODEL_PATH"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo ""
    echo "1) Single GPU (0.85 util, ~131K context)"
    echo "   - Leaves the other GPU free"
    echo ""
    echo "2) Single GPU (0.40 util, ~65K context)"
    echo "   - Low footprint, run alongside other models"
    echo ""
    echo "3) Dual GPU (0.85 util, ~262K context)"
    echo "   - Max context, both GPUs"
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
            GPU_UTIL=0.85
            MAX_MODEL_LEN=131072
            GPU_LABEL="Single GPU $gpu_id (0.85 util)"
            ;;
        2)
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
            GPU_UTIL=0.40
            MAX_MODEL_LEN=65536
            GPU_LABEL="Single GPU $gpu_id (0.40 util)"
            ;;
        3)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            GPU_UTIL=0.85
            MAX_MODEL_LEN=262144
            GPU_LABEL="Dual GPU (0.85 util)"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Configuration:"
    echo "  Backend:     vLLM  (conda env: vllm)"
    echo "  Model:       OmniCoder-9B"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Context:     $MAX_MODEL_LEN tokens"
    echo "  Port:        $PORT_OMNICODER"
    echo ""

    # Activate env with vLLM 0.17.1+ (cu130) which supports Qwen3_5ForConditionalGeneration
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate vllm

    env $NCCL_ENV \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    nohup vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port $PORT_OMNICODER \
        --tensor-parallel-size $TP_SIZE \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --dtype auto \
        --trust-remote-code \
        --served-model-name "OmniCoder-9B" \
        --disable-custom-all-reduce \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE_OMNICODER"
    echo "Server started with PID $(cat $PID_FILE_OMNICODER)"
    echo "Logs: $LOG_FILE"
    echo ""
    wait_for_vllm $PORT_OMNICODER $PID_FILE_OMNICODER $LOG_FILE

elif [ "$model_choice" == "7" ]; then

# ─────────────────────────────────────────────
# QWEN3.5-35B-A3B AGGRESSIVE (MoE, BF16 GGUF, llama.cpp)
# ─────────────────────────────────────────────

    if [ -f "$PID_FILE_AGG35B" ]; then
        PID=$(cat "$PID_FILE_AGG35B")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Aggressive-35B server already running with PID $PID"
            exit 1
        fi
    fi

    source ~/.cuda13_env

    MODEL_PATH=~/models/qwen3/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-BF16/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-BF16.gguf
    LLAMA_BIN=~/llama.cpp/build/bin/llama-server
    LOG_FILE=$LOG_DIR/aggressive-35b-llama.log

    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model not found: $MODEL_PATH"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo "  (Model is 65GB BF16 — fits on a single 96GB GPU)"
    echo ""
    echo "1) Single GPU (~31K context)"
    echo "   - 65GB model on one 96GB GPU, leaves other free"
    echo ""
    echo "2) Dual GPU (~131K context)"
    echo "   - Model split across both GPUs"
    echo ""
    echo "3) Dual GPU (~262K context)"
    echo "   - Maximum context, high VRAM usage"
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
                    echo "Starting on GPU $gpu_id (GPU $OTHER_GPU remains free)..."
                    ;;
                *)
                    echo "Invalid GPU. Exiting."
                    exit 1
                    ;;
            esac
            GPU_LAYERS=999
            TENSOR_SPLIT=""
            CTX_SIZE=32768
            GPU_LABEL="Single GPU $gpu_id, 32K ctx"
            ;;
        2)
            CUDA_DEVICES="0,1"
            GPU_LAYERS=999
            TENSOR_SPLIT="--tensor-split 50,50"
            CTX_SIZE=131072
            GPU_LABEL="Dual GPU 50/50, 131K ctx"
            ;;
        3)
            CUDA_DEVICES="0,1"
            GPU_LAYERS=999
            TENSOR_SPLIT="--tensor-split 50,50"
            CTX_SIZE=262144
            GPU_LABEL="Dual GPU 50/50, 262K ctx"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Choose sampling preset:"
    echo ""
    echo "1) Thinking / General   temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5"
    echo "2) Thinking / Coding    temp=0.6, top_p=0.95, top_k=20, presence_penalty=0.0"
    echo "3) Non-Think / General  temp=0.7, top_p=0.8,  top_k=20, presence_penalty=1.5"
    echo "4) Non-Think / Reason   temp=1.0, top_p=1.0,  top_k=40, presence_penalty=2.0"
    echo ""
    read -p "Enter choice [1-4]: " sampling_choice

    case $sampling_choice in
        1)
            SAMPLING_LABEL="Thinking/General"
            TEMP=1.0; TOP_P=0.95; TOP_K=20; PRESENCE=1.5
            ;;
        2)
            SAMPLING_LABEL="Thinking/Coding"
            TEMP=0.6; TOP_P=0.95; TOP_K=20; PRESENCE=0.0
            ;;
        3)
            SAMPLING_LABEL="NonThink/General"
            TEMP=0.7; TOP_P=0.8; TOP_K=20; PRESENCE=1.5
            ;;
        4)
            SAMPLING_LABEL="NonThink/Reasoning"
            TEMP=1.0; TOP_P=1.0; TOP_K=40; PRESENCE=2.0
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Configuration:"
    echo "  Backend:     llama.cpp"
    echo "  Model:       Qwen3.5-35B-A3B Aggressive (BF16, 65GB, MoE 256E 8+1)"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Sampling:    $SAMPLING_LABEL  (temp=$TEMP, top_p=$TOP_P, top_k=$TOP_K, presence=$PRESENCE)"
    echo "  Context:     $CTX_SIZE tokens"
    echo "  Port:        $PORT_AGG35B"
    echo ""

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup "$LLAMA_BIN" \
        --model "$MODEL_PATH" \
        --jinja \
        --n-gpu-layers $GPU_LAYERS \
        $TENSOR_SPLIT \
        --ctx-size $CTX_SIZE \
        --flash-attn on \
        --cache-type-k f16 \
        --cache-type-v f16 \
        --temp $TEMP \
        --top-p $TOP_P \
        --top-k $TOP_K \
        --host 127.0.0.1 \
        --port $PORT_AGG35B \
        --threads 16 \
        --batch-size 4096 \
        --ubatch-size 1024 \
        --poll 100 \
        --mlock \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE_AGG35B"
    echo "llama.cpp" > "$LOG_DIR/aggressive-35b-server.backend"
    echo "Server started with PID $(cat $PID_FILE_AGG35B)"
    echo "Logs: $LOG_FILE"
    echo ""
    echo "Waiting for server to be ready..."
    for i in {1..120}; do
        if curl -s http://127.0.0.1:$PORT_AGG35B/health > /dev/null 2>&1; then
            echo "✓ Server is ready!"
            echo ""
            echo "Test with:"
            echo "  curl http://127.0.0.1:$PORT_AGG35B/v1/chat/completions \\"
            echo "    -H 'Content-Type: application/json' \\"
            echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
            exit 0
        fi
        sleep 1
    done
    echo "Server may still be starting. Check logs: tail -f $LOG_FILE"

elif [ "$model_choice" == "8" ]; then

# ─────────────────────────────────────────────
# QWEN3.5-27B AGGRESSIVE (Dense, BF16 GGUF, llama.cpp)
# ─────────────────────────────────────────────

    if [ -f "$PID_FILE_AGG27B" ]; then
        PID=$(cat "$PID_FILE_AGG27B")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Aggressive-27B server already running with PID $PID"
            exit 1
        fi
    fi

    source ~/.cuda13_env

    MODEL_PATH=~/models/qwen3/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-BF16/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf
    LLAMA_BIN=~/llama.cpp/build/bin/llama-server
    LOG_FILE=$LOG_DIR/aggressive-27b-llama.log

    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model not found: $MODEL_PATH"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo "  (Model is 51GB BF16 — fits on single GPU or split across two)"
    echo ""
    echo "1) Single GPU (full offload, ~131K context)"
    echo "   - 51GB model on one 96GB GPU, leaves other free"
    echo ""
    echo "2) Dual GPU (split, ~262K context)"
    echo "   - Maximum context across both GPUs"
    echo ""
    read -p "Enter choice [1-2]: " gpu_choice

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
                    echo "Starting on GPU $gpu_id (GPU $OTHER_GPU remains free)..."
                    ;;
                *)
                    echo "Invalid GPU. Exiting."
                    exit 1
                    ;;
            esac
            GPU_LAYERS=999
            TENSOR_SPLIT=""
            CTX_SIZE=131072
            GPU_LABEL="Single GPU $gpu_id, 131K ctx"
            ;;
        2)
            CUDA_DEVICES="0,1"
            GPU_LAYERS=999
            TENSOR_SPLIT="--tensor-split 50,50"
            CTX_SIZE=262144
            GPU_LABEL="Dual GPU 50/50, 262K ctx"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Choose sampling preset:"
    echo ""
    echo "1) Thinking / Default   temp=0.6, top_p=0.95, top_k=20"
    echo "2) Non-Think / General  temp=0.7, top_p=0.8,  top_k=20"
    echo ""
    read -p "Enter choice [1-2]: " sampling_choice

    case $sampling_choice in
        1)
            SAMPLING_LABEL="Thinking/Default"
            TEMP=0.6; TOP_P=0.95; TOP_K=20
            ;;
        2)
            SAMPLING_LABEL="NonThink/General"
            TEMP=0.7; TOP_P=0.8; TOP_K=20
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Configuration:"
    echo "  Backend:     llama.cpp"
    echo "  Model:       Qwen3.5-27B Aggressive (BF16, 51GB, dense)"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Sampling:    $SAMPLING_LABEL  (temp=$TEMP, top_p=$TOP_P, top_k=$TOP_K)"
    echo "  Context:     $CTX_SIZE tokens"
    echo "  Port:        $PORT_AGG27B"
    echo ""

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup "$LLAMA_BIN" \
        --model "$MODEL_PATH" \
        --jinja \
        --n-gpu-layers $GPU_LAYERS \
        $TENSOR_SPLIT \
        --ctx-size $CTX_SIZE \
        --flash-attn on \
        --cache-type-k f16 \
        --cache-type-v f16 \
        --temp $TEMP \
        --top-p $TOP_P \
        --top-k $TOP_K \
        --host 127.0.0.1 \
        --port $PORT_AGG27B \
        --threads 16 \
        --batch-size 4096 \
        --ubatch-size 1024 \
        --poll 100 \
        --mlock \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE_AGG27B"
    echo "llama.cpp" > "$LOG_DIR/aggressive-27b-server.backend"
    echo "Server started with PID $(cat $PID_FILE_AGG27B)"
    echo "Logs: $LOG_FILE"
    echo ""
    echo "Waiting for server to be ready..."
    for i in {1..120}; do
        if curl -s http://127.0.0.1:$PORT_AGG27B/health > /dev/null 2>&1; then
            echo "✓ Server is ready!"
            echo ""
            echo "Test with:"
            echo "  curl http://127.0.0.1:$PORT_AGG27B/v1/chat/completions \\"
            echo "    -H 'Content-Type: application/json' \\"
            echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
            exit 0
        fi
        sleep 1
    done
    echo "Server may still be starting. Check logs: tail -f $LOG_FILE"

elif [ "$model_choice" == "9" ]; then

# ─────────────────────────────────────────────
# QWEN3.6-35B-A3B-FP8 (MoE, vLLM)
# ─────────────────────────────────────────────

    if [ -f "$PID_FILE_Q36_35B" ]; then
        PID=$(cat "$PID_FILE_Q36_35B")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Qwen3.6-35B-A3B server already running with PID $PID"
            exit 1
        fi
    fi

    MODEL_PATH=~/models/qwen3/Qwen3.6-35B-A3B-FP8
    LOG_FILE=$LOG_DIR/qwen36-35b-vllm.log

    if [ ! -d "$MODEL_PATH" ]; then
        echo ""
        echo "Model not found at $MODEL_PATH"
        echo "Download with:"
        echo "  huggingface-cli download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir $MODEL_PATH"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo "  (Model is 37.5GB FP8 MoE — fits comfortably on a single 96GB GPU)"
    echo ""
    echo "1) Single GPU (0.75 util, ~262K context)"
    echo "   - Leaves the other GPU free"
    echo ""
    echo "2) Dual GPU - Solo (0.85 util, ~262K context)"
    echo "   - Split across both GPUs for max KV cache"
    echo "   - Other models should NOT be running"
    echo ""
    echo "3) Dual GPU - Shared mode (0.55 util, ~131K context)"
    echo "   - Leaves room for Coder in shared mode (0.60 util)"
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
            MEM_FRAC=0.75
            MAX_MODEL_LEN=262144
            GPU_LABEL="Single GPU $gpu_id (0.75 util)"
            ;;
        2)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.85
            MAX_MODEL_LEN=262144
            GPU_LABEL="Dual GPU - Solo (0.85 util)"
            ;;
        3)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.55
            MAX_MODEL_LEN=131072
            GPU_LABEL="Dual GPU - Shared mode (0.55 util)"
            echo ""
            echo "Note: Make sure Coder is running in shared mode (0.60 util) on port $PORT_CODER"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Choose mode:"
    echo ""
    echo "1) Tools             tool calling + reasoning  (use with Claude Code)"
    echo "2) Tools + MTP       tools + speculative decoding (faster, ~2x tokens/s)"
    echo "3) Text only         reasoning only, no tool calling"
    echo "                     (skips vision encoder, frees VRAM for KV cache)"
    echo ""
    read -p "Enter choice [1-3]: " mode_choice

    case $mode_choice in
        1)
            MODE_FLAGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
            MODE_SPECULATIVE=""
            MODE_LABEL="Tools"
            ;;
        2)
            MODE_FLAGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
            echo ""
            echo "Choose MTP tokens:"
            echo "  1) 2 tokens (default, recommended)"
            echo "  2) 3 tokens (multi-user)"
            read -p "Enter choice [1-2]: " mtp_choice
            case $mtp_choice in
                2) MODE_SPECULATIVE='{"method":"mtp","num_speculative_tokens":3}' ;;
                *) MODE_SPECULATIVE='{"method":"mtp","num_speculative_tokens":2}' ;;
            esac
            MODE_LABEL="Tools + MTP"
            ;;
        3)
            MODE_FLAGS=(--reasoning-parser qwen3 --language-model-only)
            MODE_SPECULATIVE=""
            MODE_LABEL="Text only"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Choose sampling preset:"
    echo ""
    echo "1) Thinking / General   temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo "2) Thinking / Coding    temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,  repetition_penalty=1.0"
    echo "3) Instruct / General   temp=0.7, top_p=0.8,  top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo "4) Instruct / Reasoning temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo ""
    read -p "Enter choice [1-4]: " sampling_choice

    case $sampling_choice in
        1)
            SAMPLING_LABEL="Thinking/General"
            SAMPLING_PARAMS="temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=true
            ;;
        2)
            SAMPLING_LABEL="Thinking/Coding"
            SAMPLING_PARAMS="temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0"
            ENABLE_THINKING=true
            ;;
        3)
            SAMPLING_LABEL="Instruct/General"
            SAMPLING_PARAMS="temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=false
            ;;
        4)
            SAMPLING_LABEL="Instruct/Reasoning"
            SAMPLING_PARAMS="temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=false
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Configuration:"
    echo "  Backend:     vLLM  (conda env: vllm)"
    echo "  Model:       Qwen3.6-35B-A3B-FP8 (MoE, 37.5GB)"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Mode:        $MODE_LABEL"
    echo "  Sampling:    $SAMPLING_LABEL  ($SAMPLING_PARAMS)"
    echo "  Context:     $MAX_MODEL_LEN tokens"
    echo "  Port:        $PORT_Q36_35B"
    echo ""

    SPEC_ARGS=()
    if [ -n "$MODE_SPECULATIVE" ]; then
        SPEC_ARGS=("--speculative-config" "$MODE_SPECULATIVE")
    fi

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate vllm

    env $NCCL_ENV \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    nohup vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port $PORT_Q36_35B \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $MEM_FRAC \
        --max-model-len $MAX_MODEL_LEN \
        --served-model-name "Qwen3.6-35B-A3B" \
        --dtype auto \
        --trust-remote-code \
        --attention-backend FLASH_ATTN \
        --disable-custom-all-reduce \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        "${MODE_FLAGS[@]}" \
        "${SPEC_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE_Q36_35B"
    echo "vllm" > ~/qwen-service/qwen36-35b-server.backend
    echo "Server started with PID $(cat $PID_FILE_Q36_35B)"
    echo "Logs: $LOG_FILE"
    echo ""
    wait_for_vllm_then_continue $PORT_Q36_35B $PID_FILE_Q36_35B $LOG_FILE

    echo "Running warmup request (Triton kernel compilation, ~50s)..."
    WARMUP_START=$SECONDS
    curl -s http://127.0.0.1:$PORT_Q36_35B/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"Qwen3.6-35B-A3B\",
            \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
            \"max_tokens\": 5,
            \"chat_template_kwargs\": {\"enable_thinking\": $ENABLE_THINKING}
        }" > /dev/null 2>&1
    WARMUP_ELAPSED=$(( SECONDS - WARMUP_START ))
    echo "✓ Warmup done (${WARMUP_ELAPSED}s). Server is ready for real requests."
    echo ""
    echo "Test with:"
    echo "  curl http://127.0.0.1:$PORT_Q36_35B/v1/models | python3 -m json.tool"
    echo ""

elif [ "$model_choice" == "10" ]; then

# ─────────────────────────────────────────────
# QWEN3.6-27B-FP8 (Dense, vLLM)
# ─────────────────────────────────────────────

    if [ -f "$PID_FILE_Q36_27B" ]; then
        PID=$(cat "$PID_FILE_Q36_27B")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Qwen3.6-27B server already running with PID $PID"
            exit 1
        fi
    fi

    MODEL_PATH=~/models/qwen3/Qwen3.6-27B-FP8
    LOG_FILE=$LOG_DIR/qwen36-27b-vllm.log

    if [ ! -d "$MODEL_PATH" ]; then
        echo ""
        echo "Model not found at $MODEL_PATH"
        echo "Download with:"
        echo "  huggingface-cli download Qwen/Qwen3.6-27B-FP8 --local-dir $MODEL_PATH"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo "  (Model is 31GB FP8 — fits easily on a single 96GB GPU)"
    echo ""
    echo "1) Single GPU (0.80 util, ~262K context)"
    echo "   - Leaves the other GPU free"
    echo ""
    echo "2) Dual GPU - Solo (0.85 util, ~262K context)"
    echo "   - Split across both GPUs for max KV cache"
    echo "   - Other models should NOT be running"
    echo ""
    echo "3) Dual GPU - Shared mode (0.55 util, ~131K context)"
    echo "   - Leaves room for Coder in shared mode (0.60 util)"
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
            MEM_FRAC=0.80
            MAX_MODEL_LEN=262144
            GPU_LABEL="Single GPU $gpu_id (0.80 util)"
            ;;
        2)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.85
            MAX_MODEL_LEN=262144
            GPU_LABEL="Dual GPU - Solo (0.85 util)"
            ;;
        3)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.55
            MAX_MODEL_LEN=131072
            GPU_LABEL="Dual GPU - Shared mode (0.55 util)"
            echo ""
            echo "Note: Make sure Coder is running in shared mode (0.60 util) on port $PORT_CODER"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Choose mode:"
    echo ""
    echo "1) Tools             tool calling + reasoning  (use with Claude Code)"
    echo "2) Tools + MTP       tools + speculative decoding (faster, ~2x tokens/s)"
    echo "3) Text only         reasoning only, no tool calling"
    echo "                     (skips vision encoder, frees VRAM for KV cache)"
    echo ""
    read -p "Enter choice [1-3]: " mode_choice

    case $mode_choice in
        1)
            MODE_FLAGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
            MODE_SPECULATIVE=""
            MODE_LABEL="Tools"
            ;;
        2)
            MODE_FLAGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
            echo ""
            echo "Choose MTP tokens:"
            echo "  1) 2 tokens (default, recommended)"
            echo "  2) 3 tokens (multi-user)"
            read -p "Enter choice [1-2]: " mtp_choice
            case $mtp_choice in
                2) MODE_SPECULATIVE='{"method":"mtp","num_speculative_tokens":3}' ;;
                *) MODE_SPECULATIVE='{"method":"mtp","num_speculative_tokens":2}' ;;
            esac
            MODE_LABEL="Tools + MTP"
            ;;
        3)
            MODE_FLAGS=(--reasoning-parser qwen3 --language-model-only)
            MODE_SPECULATIVE=""
            MODE_LABEL="Text only"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Choose sampling preset:"
    echo ""
    echo "1) Thinking / General   temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo "2) Thinking / Coding    temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,  repetition_penalty=1.0"
    echo "3) Instruct / General   temp=0.7, top_p=0.8,  top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo "4) Instruct / Reasoning temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo ""
    read -p "Enter choice [1-4]: " sampling_choice

    case $sampling_choice in
        1)
            SAMPLING_LABEL="Thinking/General"
            SAMPLING_PARAMS="temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=true
            ;;
        2)
            SAMPLING_LABEL="Thinking/Coding"
            SAMPLING_PARAMS="temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0"
            ENABLE_THINKING=true
            ;;
        3)
            SAMPLING_LABEL="Instruct/General"
            SAMPLING_PARAMS="temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=false
            ;;
        4)
            SAMPLING_LABEL="Instruct/Reasoning"
            SAMPLING_PARAMS="temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0"
            ENABLE_THINKING=false
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Configuration:"
    echo "  Backend:     vLLM  (conda env: vllm)"
    echo "  Model:       Qwen3.6-27B-FP8 (dense, 31GB)"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Mode:        $MODE_LABEL"
    echo "  Sampling:    $SAMPLING_LABEL  ($SAMPLING_PARAMS)"
    echo "  Context:     $MAX_MODEL_LEN tokens"
    echo "  Port:        $PORT_Q36_27B"
    echo ""

    SPEC_ARGS=()
    if [ -n "$MODE_SPECULATIVE" ]; then
        SPEC_ARGS=("--speculative-config" "$MODE_SPECULATIVE")
    fi

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate vllm

    env $NCCL_ENV \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    nohup vllm serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port $PORT_Q36_27B \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $MEM_FRAC \
        --max-model-len $MAX_MODEL_LEN \
        --served-model-name "Qwen3.6-27B" \
        --dtype auto \
        --trust-remote-code \
        --attention-backend FLASH_ATTN \
        --disable-custom-all-reduce \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        "${MODE_FLAGS[@]}" \
        "${SPEC_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE_Q36_27B"
    echo "vllm" > ~/qwen-service/qwen36-27b-server.backend
    echo "Server started with PID $(cat $PID_FILE_Q36_27B)"
    echo "Logs: $LOG_FILE"
    echo ""
    wait_for_vllm_then_continue $PORT_Q36_27B $PID_FILE_Q36_27B $LOG_FILE

    echo "Running warmup request (Triton kernel compilation, ~50s)..."
    WARMUP_START=$SECONDS
    curl -s http://127.0.0.1:$PORT_Q36_27B/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"Qwen3.6-27B\",
            \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
            \"max_tokens\": 5,
            \"chat_template_kwargs\": {\"enable_thinking\": $ENABLE_THINKING}
        }" > /dev/null 2>&1
    WARMUP_ELAPSED=$(( SECONDS - WARMUP_START ))
    echo "✓ Warmup done (${WARMUP_ELAPSED}s). Server is ready for real requests."
    echo ""
    echo "Test with:"
    echo "  curl http://127.0.0.1:$PORT_Q36_27B/v1/models | python3 -m json.tool"
    echo ""

else
    echo "Invalid choice. Exiting."
    exit 1
fi