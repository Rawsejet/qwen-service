#!/bin/bash
# Start Qwen3 Model Server - Interactive Mode Selection
# Backends: vLLM, llama.cpp
# Port is selected interactively (defaults in models.conf)

LOG_DIR=~/qwen-service

# Source shared model registry
source "$LOG_DIR/models.conf"

BACKEND_FILE=$LOG_DIR/qwen-server.backend

# NCCL tuning for RTX PRO 6000 Blackwell (SM120, PCIe PHB topology, no NVLink):
# P2P/CUMEM and P2P/IPC hang on SM120 - working transport is SHM/direct/direct
# NCCL_P2P_DISABLE=1 + NCCL_CUMEM_ENABLE=0 forces SHM transport which works
# --disable-custom-all-reduce required: vLLM's custom all-reduce IPC also hangs on SM120
# vLLM 0.24+: DeepGemm became the default FP8 GEMM backend and crashes with
# "Unknown recipe" on Qwen FP8 checkpoints (scale_fmt is not ue8m0) - disable it
# vLLM 0.24+: hybrid GDN models (Qwen3.6) refuse to start if max_num_seqs exceeds
# the Mamba cache blocks - vllm serve calls below pass --max-num-seqs 256
NCCL_ENV="NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_IB_DISABLE=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_ENGINE_CORE_STARTUP_TIMEOUT=300 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

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
    printf "%d) %-22s\n" "$i" "${MODEL_NAMES[$i]}"
    echo "   ${MODEL_DESCS[$i]}"
    echo ""
done
read -p "Enter choice [1-$MODEL_COUNT]: " model_choice

# ─────────────────────────────────────────────
# Port selection
# ─────────────────────────────────────────────
PID_FILE=$LOG_DIR/${MODEL_PIDS[$model_choice]}
PORT_FILE="${PID_FILE%.pid}.port"
DEFAULT_PORT=${MODEL_DEFAULT_PORTS[$model_choice]}

echo ""
read -p "Port [default: $DEFAULT_PORT]: " port_input
PORT=${port_input:-$DEFAULT_PORT}


# ─────────────────────────────────────────────
# Host binding (local vs LAN)
# ─────────────────────────────────────────────
echo ""
echo "Bind address:"
echo "  1) Local only (127.0.0.1)   — reachable from this machine only"
echo "  2) LAN (0.0.0.0)            — reachable from other devices on your network"
read -p "Enter choice [default: 1]: " host_input
case "${host_input:-1}" in
    2)
        HOST=0.0.0.0
        HOST_LABEL="LAN — reachable at http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
        echo ""
        echo "⚠️  Binding 0.0.0.0 exposes this server to your LAN with no auth. Trusted networks only."
        ;;
    *)
        HOST=127.0.0.1
        HOST_LABEL="local only"
        ;;
esac

# ─────────────────────────────────────────────
# QWEN3.8-27B (Dense, BF16, vision + thinking, vLLM)  [FLAGSHIP]
# ─────────────────────────────────────────────
if [ "$model_choice" == "1" ]; then

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Qwen3.8-27B server already running with PID $PID"
            exit 1
        fi
    fi

    MODEL_PATH=~/models/qwen3/Qwen3.8-27B
    LOG_FILE=$LOG_DIR/qwen38-27b-vllm.log

    if [ ! -d "$MODEL_PATH" ]; then
        echo ""
        echo "Model not found at $MODEL_PATH"
        echo "Download with:"
        echo "  huggingface-cli download Qwen/Qwen3.8-27B --local-dir $MODEL_PATH"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo "  (Model is ~54GB BF16 — same qwen3_5 hybrid arch as Qwen3.5-27B)"
    echo ""
    echo "1) Single GPU (0.85 mem, ~131K context)"
    echo "   - Leaves the other GPU free"
    echo ""
    echo "2) Dual GPU - Solo (0.85 mem, 262K native context)"
    echo "   - Split across both GPUs for max native context"
    echo "   - Other models should NOT be running"
    echo ""
    echo "3) Dual GPU - Shared mode (0.55 mem, ~65K context)"
    echo "   - Leaves room for another model on both GPUs"
    echo ""
    echo "4) Dual GPU - 512K long context (0.85 mem, YaRN 2x)"
    echo "   - Extends 262K native -> 524288 via YaRN"
    echo "   - Single-user (~2.9x concurrency); slight quality drift on short prompts"
    echo ""
    echo "5) Dual GPU - 1M max context (0.85 mem, YaRN 4x)"
    echo "   - Extends 262K native -> 1048576 via YaRN (model-card ceiling)"
    echo "   - Single-user (~1.5x concurrency); more quality drift on short prompts"
    echo ""
    read -p "Enter choice [1-5]: " gpu_choice

    # YaRN rope override, only set for the long-context modes (4/5)
    ROPE_ARGS=()
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
            MAX_MODEL_LEN=131072
            GPU_LABEL="Single GPU $gpu_id"
            ;;
        2)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.85
            MAX_MODEL_LEN=262144
            GPU_LABEL="Dual GPU - Solo"
            ;;
        3)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.55
            MAX_MODEL_LEN=65536
            GPU_LABEL="Dual GPU - Shared mode"
            ;;
        4)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.85
            MAX_MODEL_LEN=524288
            ROPE_ARGS=(--hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144}}}')
            GPU_LABEL="Dual GPU - 512K (YaRN 2x)"
            ;;
        5)
            CUDA_DEVICES="0,1"
            TP_SIZE=2
            MEM_FRAC=0.85
            MAX_MODEL_LEN=1048576
            ROPE_ARGS=(--hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}}}')
            GPU_LABEL="Dual GPU - 1M (YaRN 4x)"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Choose weight precision:"
    echo ""
    echo "1) BF16          full quality (~28 tok/s single-stream baseline)"
    echo "2) FP8 dynamic   on-the-fly FP8 quant — ~2x faster + half VRAM, tiny quality cost"
    echo "                 (benchmarked 2026-08-25: FP8 + Mode 2 MTP = 75 tok/s single / 458 batch;"
    echo "                  add dual-GPU for ~104 tok/s. Pick this + Mode 2 for the fast config.)"
    echo ""
    read -p "Enter choice [default: 1]: " quant_choice
    case "${quant_choice:-1}" in
        2)
            QUANT_ARGS=(--quantization fp8)
            QUANT_LABEL="FP8 dynamic"
            ;;
        *)
            QUANT_ARGS=()
            QUANT_LABEL="BF16"
            ;;
    esac

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
    echo "Choose sampling preset (Qwen3.8 model-card values):"
    echo ""
    echo "1) Thinking / General   temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,  repetition_penalty=1.0"
    echo "2) Thinking / Coding    temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,  repetition_penalty=1.0"
    echo "3) Instruct / General   temp=0.7, top_p=0.8,  top_k=20, min_p=0.0, presence_penalty=1.5,  repetition_penalty=1.0"
    echo "4) Instruct / Reasoning temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,  repetition_penalty=1.0"
    echo ""
    read -p "Enter choice [1-4]: " sampling_choice

    case $sampling_choice in
        1)
            SAMPLING_LABEL="Thinking/General"
            SAMPLING_PARAMS="temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0"
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
            SAMPLING_PARAMS="temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0"
            ENABLE_THINKING=false
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    # Save preset for reference
    cat > ~/qwen-service/qwen38-27b-sampling.conf << CONF
SAMPLING_LABEL=$SAMPLING_LABEL
SAMPLING_PARAMS=$SAMPLING_PARAMS
ENABLE_THINKING=$ENABLE_THINKING
CONF

    echo ""
    echo "Configuration:"
    echo "  Backend:     vLLM  (conda env: vllm)"
    echo "  Model:       Qwen3.8-27B (dense, vision, ~54GB BF16)"
    echo "  Precision:   $QUANT_LABEL"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Host:        $HOST  ($HOST_LABEL)"
    echo "  Mode:        $MODE_LABEL"
    echo "  Sampling:    $SAMPLING_LABEL  ($SAMPLING_PARAMS)"
    if [ ${#ROPE_ARGS[@]} -gt 0 ]; then
        echo "  Context:     $MAX_MODEL_LEN tokens  (YaRN-extended over 262K native — expect slight quality drift on short prompts)"
    else
        echo "  Context:     $MAX_MODEL_LEN tokens  (262K native, YaRN-extensible to 1M via options 4/5)"
    fi
    echo "  Port:        $PORT"
    echo ""
    echo "  Note: Thinking is ON by default. Disable per request with"
    echo "        chat_template_kwargs={\"enable_thinking\": false}."
    echo "        Reasoning depth: reasoning_effort = xhigh (default) | medium | low."
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
        --host $HOST \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $MEM_FRAC \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs 256 \
        "${ROPE_ARGS[@]}" \
        "${QUANT_ARGS[@]}" \
        --served-model-name "Qwen3.8-27B" \
        --dtype auto \
        --trust-remote-code \
        --attention-backend FLASH_ATTN \
        --disable-custom-all-reduce \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        "${MODE_FLAGS[@]}" \
        "${SPEC_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "$PORT" > "$PORT_FILE"
    echo "vllm" > ~/qwen-service/qwen38-27b-server.backend
    echo "Server started with PID $(cat $PID_FILE)"
    echo "Logs: $LOG_FILE"
    echo ""
    wait_for_vllm_then_continue $PORT $PID_FILE $LOG_FILE

    echo "Running warmup request (Triton kernel compilation, ~50s)..."
    WARMUP_START=$SECONDS
    curl -s http://127.0.0.1:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"Qwen3.8-27B\",
            \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
            \"max_tokens\": 5,
            \"chat_template_kwargs\": {\"enable_thinking\": $ENABLE_THINKING}
        }" > /dev/null 2>&1
    WARMUP_ELAPSED=$(( SECONDS - WARMUP_START ))
    echo "✓ Warmup done (${WARMUP_ELAPSED}s). Server is ready for real requests."
    echo ""
    echo "Test with:"
    echo "  curl http://127.0.0.1:$PORT/v1/models | python3 -m json.tool"
    echo ""

elif [ "$model_choice" == "2" ]; then

# ─────────────────────────────────────────────
# QWEN3.6-35B-A3B-FP8 (MoE, vLLM)
# ─────────────────────────────────────────────

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
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
    echo "   - Max KV cache; likely faster decode (27B measured +53% with TP=2)"
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
            echo "Note: Make sure Coder is running in shared mode (0.60 util)"
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
    echo "  Port:        $PORT"
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
        --host $HOST \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $MEM_FRAC \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs 256 \
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

    echo $! > "$PID_FILE"
    echo "$PORT" > "$PORT_FILE"
    echo "vllm" > ~/qwen-service/qwen36-35b-server.backend
    echo "Server started with PID $(cat $PID_FILE)"
    echo "Logs: $LOG_FILE"
    echo ""
    wait_for_vllm_then_continue $PORT $PID_FILE $LOG_FILE

    echo "Running warmup request (Triton kernel compilation, ~50s)..."
    WARMUP_START=$SECONDS
    curl -s http://127.0.0.1:$PORT/v1/chat/completions \
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
    echo "  curl http://127.0.0.1:$PORT/v1/models | python3 -m json.tool"
    echo ""

elif [ "$model_choice" == "3" ]; then

# ─────────────────────────────────────────────
# QWEN3.5-35B-A3B AGGRESSIVE (MoE, BF16 GGUF, llama.cpp)
# ─────────────────────────────────────────────

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Aggressive-35B server already running with PID $PID"
            exit 1
        fi
    fi

    source ~/.cuda13_env

    MODEL_PATH=~/models/qwen3/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-BF16/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-BF16.gguf
    LLAMA_BIN=~/llama.cpp/build/bin/llama-server
    LOG_FILE=$LOG_DIR/aggressive-35b-llama.log
    # Patched copy of this GGUF's embedded chat template: merges all system
    # messages into the leading system block instead of raising, since Claude
    # Code >=2.1.154 sends system-role entries in messages[] alongside the
    # top-level system field. Only model 7 uses this file.
    TEMPLATE_FILE=$LOG_DIR/aggressive-35b-template.jinja

    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model not found: $MODEL_PATH"
        exit 1
    fi

    if [ ! -f "$TEMPLATE_FILE" ]; then
        echo "Chat template not found: $TEMPLATE_FILE"
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
    echo "  Port:        $PORT"
    echo ""

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup "$LLAMA_BIN" \
        --model "$MODEL_PATH" \
        --jinja \
        --chat-template-file "$TEMPLATE_FILE" \
        --n-gpu-layers $GPU_LAYERS \
        $TENSOR_SPLIT \
        --ctx-size $CTX_SIZE \
        --flash-attn on \
        --cache-type-k f16 \
        --cache-type-v f16 \
        --temp $TEMP \
        --top-p $TOP_P \
        --top-k $TOP_K \
        --host $HOST \
        --port $PORT \
        --threads 16 \
        --batch-size 4096 \
        --ubatch-size 1024 \
        --poll 100 \
        --mlock \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "$PORT" > "$PORT_FILE"
    echo "llama.cpp" > "$LOG_DIR/aggressive-35b-server.backend"
    echo "Server started with PID $(cat $PID_FILE)"
    echo "Logs: $LOG_FILE"
    echo ""
    echo "Waiting for server to be ready..."
    for i in {1..120}; do
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            echo "✓ Server is ready!"
            echo ""
            echo "Test with:"
            echo "  curl http://127.0.0.1:$PORT/v1/chat/completions \\"
            echo "    -H 'Content-Type: application/json' \\"
            echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
            exit 0
        fi
        sleep 1
    done
    echo "Server may still be starting. Check logs: tail -f $LOG_FILE"

elif [ "$model_choice" == "4" ]; then

# ─────────────────────────────────────────────
# QWEN3.6-27B AGGRESSIVE (Dense, Q8 GGUF, llama.cpp)
# ─────────────────────────────────────────────

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Aggressive-27B server already running with PID $PID"
            exit 1
        fi
    fi

    source ~/.cuda13_env

    MODEL_PATH="$HOME/models/qwen3/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive /Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
    LLAMA_BIN=~/llama.cpp/build/bin/llama-server
    LOG_FILE=$LOG_DIR/aggressive-27b-llama.log
    # Patched copy of this GGUF's embedded chat template: merges all system
    # messages into the leading system block instead of raising, since Claude
    # Code >=2.1.154 sends system-role entries in messages[] alongside the
    # top-level system field. Only model 8 uses this file.
    TEMPLATE_FILE=$LOG_DIR/aggressive-27b-template.jinja

    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model not found: $MODEL_PATH"
        exit 1
    fi

    if [ ! -f "$TEMPLATE_FILE" ]; then
        echo "Chat template not found: $TEMPLATE_FILE"
        exit 1
    fi

    echo ""
    echo "Choose GPU configuration:"
    echo "  (Model is 30GB Q8 — fits on single GPU or split across two)"
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
    echo "  Model:       Qwen3.6-27B Aggressive (Q8, 30GB, dense)"
    echo "  GPUs:        $GPU_LABEL"
    echo "  Sampling:    $SAMPLING_LABEL  (temp=$TEMP, top_p=$TOP_P, top_k=$TOP_K)"
    echo "  Context:     $CTX_SIZE tokens"
    echo "  Port:        $PORT"
    echo ""

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup "$LLAMA_BIN" \
        --model "$MODEL_PATH" \
        --jinja \
        --chat-template-file "$TEMPLATE_FILE" \
        --n-gpu-layers $GPU_LAYERS \
        $TENSOR_SPLIT \
        --ctx-size $CTX_SIZE \
        --flash-attn on \
        --cache-type-k f16 \
        --cache-type-v f16 \
        --temp $TEMP \
        --top-p $TOP_P \
        --top-k $TOP_K \
        --host $HOST \
        --port $PORT \
        --threads 16 \
        --batch-size 4096 \
        --ubatch-size 1024 \
        --poll 100 \
        --mlock \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "$PORT" > "$PORT_FILE"
    echo "llama.cpp" > "$LOG_DIR/aggressive-27b-server.backend"
    echo "Server started with PID $(cat $PID_FILE)"
    echo "Logs: $LOG_FILE"
    echo ""
    echo "Waiting for server to be ready..."
    for i in {1..120}; do
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            echo "✓ Server is ready!"
            echo ""
            echo "Test with:"
            echo "  curl http://127.0.0.1:$PORT/v1/chat/completions \\"
            echo "    -H 'Content-Type: application/json' \\"
            echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
            exit 0
        fi
        sleep 1
    done
    echo "Server may still be starting. Check logs: tail -f $LOG_FILE"

else
    echo "Invalid choice. Exiting."
    exit 1
fi