# Qwen Service

A production-quality infrastructure for managing multiple Qwen and Llama model servers locally on a dual-GPU setup (RTX PRO 6000 Blackwell). Supports vLLM and llama.cpp backends with comprehensive resource management, health monitoring, and multi-model orchestration.

## Quick Start

```bash
# Start all servers (interactive configuration)
./start-qwen.sh

# Check server status
./status-qwen.sh

# View logs
./logs-qwen.sh [-f]  # Use -f to follow logs

# Stop all servers
./stop-qwen.sh
```

## Supported Models

| Model | Backend | Port | Context |
|-------|---------|------|---------|
| Qwen3-Coder-Next | vLLM | 8085 | Variable |
| Qwen-VL-32B | vLLM | 8088 | Variable |
| Qwen-VL-4B | vLLM | 8086 | Variable |
| Qwen-27B | vLLM | 8087 | Variable |
| Qwen-122B-A10B | vLLM | 8089 | Variable |
| OmniCoder-9B | vLLM | 8090 | Variable |

**Legacy**: Qwen3-Coder-Next also available via llama.cpp (see `start-qwen-coder.sh`)

## Architecture

### Key Features

- **Multi-Backend Support**: vLLM for most models, llama.cpp for interactive/offline workflows
- **GPU-Aware Configuration**: Hardware-aware NCCL tuning for RTX Blackwell (SM120 PCIe topology)
- **Resource Management**: Dynamic swap expansion, GPU splitting, OOM prevention
- **State Management**: PID file tracking with sanity checks before restart
- **Graceful Shutdown**: Multi-process termination with port binding cleanup
- **Performance Monitoring**: Built-in health checks and status endpoints

### Hardware Tuning

- NCCL P2P disabled (`NCCL_P2P_DISABLE=1`) for SM120 stability
- Cumulative memory disabled (`NCCL_CUMEM_ENABLE=0`)
- Custom all-reduce disabled (`--disable-custom-all-reduce`)
- Automatic swap expansion for large single-GPU models (up to 80GB)

## API Endpoints

Once running, each model exposes an OpenAI-compatible API:

```
http://127.0.0.1:PORT/v1/chat/completions
```

Replace `PORT` with the model's assigned port (see table above).

## Scripts

| Script | Purpose |
|--------|---------|
| `start-qwen.sh` | Main launcher; manages all vLLM models with interactive selection |
| `start-qwen-coder.sh` | Launch Qwen3-Coder-Next via llama.cpp (interactive, Q8 only) |
| `start-qwen-interactive.sh` | Legacy alias for llama.cpp coder model |
| `stop-qwen.sh` | Graceful shutdown for all running servers; includes swap cleanup |
| `status-qwen.sh` | Check running processes and port bindings |
| `logs-qwen.sh` | Tail server logs |
| `perf-check.sh` | Monitor GPU/CPU/memory performance in real-time |

## Configuration

### models.conf

Defines port assignments and model metadata. Currently configured for 6 vLLM models:

```bash
MODEL_PORTS=( [1]=8085 [2]=8088 [3]=8086 [4]=8087 [5]=8089 [6]=8090 )
```

### qwen-27b-sampling.conf

Pre-configured sampling parameters for Qwen-27B, written at startup. Can be modified for future runs.

## Security Notes

- Swap operations use `sudo` commands; will fail if running as non-root (unless already privileged)
- `fuser -k` and `kill -9` are intentional for aggressive cleanup; review before customization
- PID-based process tracking assumes stable process naming

## Requirements

- Dual GPU setup (RTX PRO 6000 Blackwell or similar with SM120 topology)
- Linux with bash, fuser, pgrep, and sudo privileges
- vLLM and/or llama.cpp installed and in PATH
- Adequate swap space (8GB minimum, 80GB for 122B model)