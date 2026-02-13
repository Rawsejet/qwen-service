#!/bin/bash
# Start Qwen3-Coder-Next with llama.cpp - Interactive Mode Selection
# Load CUDA 13 environment
source ~/.cuda13_env
MODEL_DIR=~/models/qwen3/qwen3-coder-next
LLAMA_BIN=~/llama.cpp/build/bin/llama-server
LOG_FILE=~/qwen-service/qwen-server.log
PID_FILE=~/qwen-service/qwen-server.pid
# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Server already running with PID $PID"
        exit 1
    fi
fi
# Interactive mode selection
echo "========================================="
echo "  Qwen3 Server Configuration"
echo "========================================="
echo ""
echo "Choose model:"
echo ""
echo "1) Q8_0 (Standard 8-bit, faster)"
echo "2) UD-Q8_K_XL (Unsloth Dynamic, higher fidelity)"
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
echo "Selected model: $MODEL_NAME"
echo ""
echo "Choose GPU configuration:"
echo ""
echo "1) Single GPU"
echo "   - All model + KV cache on one GPU"
echo "   - Max context: ~170K tokens"
echo "   - Leaves the other GPU free for other tasks"
echo ""
echo "2) Dual GPU (Split across both)"
echo "   - Model split 65/35 across GPUs"
echo "   - Max context: ~256K tokens"
echo "   - Uses both GPUs"
echo ""
read -p "Enter choice [1-2]: " choice
case $choice in
    1)
        echo ""
        echo "Which GPU to use?"
        echo "  0) GPU 0"
        echo "  1) GPU 1"
        echo ""
        read -p "Enter GPU [0-1]: " gpu_choice
        case $gpu_choice in
            0|1)
                CUDA_DEVICES="$gpu_choice"
                OTHER_GPU=$(( 1 - gpu_choice ))
                echo ""
                echo "Starting with SINGLE GPU configuration (GPU $gpu_choice)..."
                echo "GPU $OTHER_GPU remains free for other tasks."
                ;;
            *)
                echo "Invalid GPU choice. Exiting."
                exit 1
                ;;
        esac
        GPU_LAYERS=999
        TENSOR_SPLIT=""
        CTX_SIZE=174762
        PORT=8085
        ;;
    2)
        echo ""
        echo "Starting with DUAL GPU configuration..."
        GPU_LAYERS=999
        TENSOR_SPLIT="--tensor-split 50,50"
        CTX_SIZE=262144
        PORT=8085
        CUDA_DEVICES="0,1"  # Use both GPUs
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  GPU Mode: $([ -z "$TENSOR_SPLIT" ] && echo "Single GPU (GPU $CUDA_DEVICES)" || echo "Dual GPU (45/55 split)")"
echo "  Context Size: $CTX_SIZE tokens"
echo "  Port: $PORT"
echo ""
# Start server in background
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
  --host 127.0.0.1 \
  --port $PORT \
  --threads $(nproc) \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Server started with PID $(cat $PID_FILE)"
echo "Logs: $LOG_FILE"
echo "Port: $PORT"
echo "Waiting for server to be ready..."
# Wait for server to be ready
for i in {1..30}; do
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
echo "Server may still be starting. Check logs with: ./logs-qwen.sh"