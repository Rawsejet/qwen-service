# Qwen3-Coder-Next Service

A managed service for running the Qwen3-Coder-Next model via llama.cpp with interactive GPU configuration.

## Quick Start

```bash
# Start the server (interactive mode selection)
./start-qwen.sh

# Check server status
./status-qwen.sh

# View logs
./logs-qwen.sh [-f]  # Use -f to follow logs

# Stop the server
./stop-qwen.sh
```

## Architecture

This service provides an interactive launcher for the Qwen3-Coder-Next language model using llama.cpp. Key features:

- **Interactive Configuration**: Choose between two model variants and GPU configurations at startup
- **Dual GPU Support**: Single GPU (170K context) or dual GPU split (256K context)
- **Health Monitoring**: Built-in health check endpoint for server readiness
- **Process Management**: PID file-based tracking with graceful shutdown

## Model Options

| Option | Description | Context |
|--------|-------------|---------|
| Q8_0 | Standard 8-bit quantization (faster) | ~170K tokens |
| UD-Q8_K_XL | Unsloth Dynamic (higher fidelity) | ~170K tokens |

## GPU Configurations

1. **Single GPU**: Uses one GPU with all model layers + KV cache; leaves the other GPU free
2. **Dual GPU**: Splits model 65/35 across both GPUs for extended context

## API Endpoint

Once running, the server exposes an OpenAI-compatible API at `http://127.0.0.1:8085/v1/chat/completions`.