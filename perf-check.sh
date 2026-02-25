#!/bin/bash
# Performance monitoring script for Qwen3 llama-server
# Usage: ./perf-check.sh [--json] [--watch]

set -euo pipefail

JSON_OUTPUT=false
WATCH_MODE=false
INTERVAL=2

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

get_performance_data() {
    # GPU metrics
    local gpu0_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
    local gpu0_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    local gpu0_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i 0)
    local gpu0_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)

    local gpu1_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1)
    local gpu1_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    local gpu1_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i 1)
    local gpu1_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 1)

    # CPU metrics
    local cpu_util=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}' | cut -d'.' -f1)
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | xargs)

    # Process metrics for llama-server
    local llama_cpu=0
    local llama_mem=0
    if pgrep -f "llama-server" > /dev/null; then
        llama_cpu=$(ps aux | grep "llama-server" | grep -v grep | awk '{print $3}')
        llama_mem=$(ps aux | grep "llama-server" | grep -v grep | awk '{print $4}')
    fi

    # Memory metrics
    local mem_used=$(free -h | grep "^Mem:" | awk '{print $3}')
    local mem_total=$(free -h | grep "^Mem:" | awk '{print $2}')
    local swap_used=$(free -h | grep "^Swap:" | awk '{print $3}')
    local swap_total=$(free -h | grep "^Swap:" | awk '{print $2}')

    # Status
    local server_status="❌ Down"
    if pgrep -f "llama-server" > /dev/null; then
        server_status="✓ Running"
    fi

    if [ "$JSON_OUTPUT" = true ]; then
        cat << EOF
{
  "timestamp": "$(date -Iseconds)",
  "server_status": "$server_status",
  "gpu": {
    "gpu0": {
      "utilization": "$gpu0_util%",
      "memory_used": "${gpu0_mem}MB",
      "temperature": "${gpu0_temp}°C",
      "power": "${gpu0_power}W"
    },
    "gpu1": {
      "utilization": "$gpu1_util%",
      "memory_used": "${gpu1_mem}MB",
      "temperature": "${gpu1_temp}°C",
      "power": "${gpu1_power}W"
    }
  },
  "cpu": {
    "utilization": "${cpu_util}%",
    "load_average": "$load_avg",
    "llama_cpu_percent": "$llama_cpu%",
    "llama_memory_percent": "$llama_mem%"
  },
  "memory": {
    "ram_used": "$mem_used",
    "ram_total": "$mem_total",
    "swap_used": "$swap_used",
    "swap_total": "$swap_total"
  }
}
EOF
    else
        clear
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║         Qwen3 Server Performance Monitor                       ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Status: $server_status"
        echo ""
        echo "┌─ GPU Performance ─────────────────────────────────────────────┐"
        echo "│                                                               │"
        printf "│ GPU 0: %3d%% util │ %5dMB mem │ %2d°C │ %5sW            │\n" "$gpu0_util" "$gpu0_mem" "$gpu0_temp" "$gpu0_power"
        printf "│ GPU 1: %3d%% util │ %5dMB mem │ %2d°C │ %5sW            │\n" "$gpu1_util" "$gpu1_mem" "$gpu1_temp" "$gpu1_power"
        echo "│                                                               │"
        echo "└───────────────────────────────────────────────────────────────┘"
        echo ""
        echo "┌─ CPU Performance ─────────────────────────────────────────────┐"
        printf "│ Overall CPU: %3d%% utilization                              │\n" "$cpu_util"
        printf "│ Load Average: %s                         │\n" "$load_avg"
        printf "│ llama-server: %3s%% CPU, %4s%% RAM                          │\n" "$llama_cpu" "$llama_mem"
        echo "│                                                               │"
        echo "└───────────────────────────────────────────────────────────────┘"
        echo ""
        echo "┌─ System Memory ───────────────────────────────────────────────┐"
        printf "│ RAM:  %s / %s                                            │\n" "$mem_used" "$mem_total"
        printf "│ Swap: %s / %s                                            │\n" "$swap_used" "$swap_total"
        echo "│                                                               │"
        echo "└───────────────────────────────────────────────────────────────┘"
        echo ""

        # Analysis
        echo "┌─ Analysis ────────────────────────────────────────────────────┐"
        if [ "$server_status" = "❌ Down" ]; then
            echo "│ ⚠️  Server is not running                                    │"
        else
            if [ "$gpu0_util" -gt 90 ]; then
                echo "│ ✓ GPU 0 well utilized (${gpu0_util}%)                           │"
            elif [ "$gpu0_util" -lt 50 ]; then
                echo "│ ⚠️  GPU 0 underutilized (${gpu0_util}%)                         │"
            else
                echo "│ ✓ GPU 0 moderate utilization (${gpu0_util}%)                   │"
            fi

            if [ "$gpu1_util" -lt 10 ] && [ "$gpu0_util" -gt 50 ]; then
                echo "│ ⚠️  GPU 1 idle - tensor-split not working optimally        │"
            fi

            if [ "$cpu_util" -gt 80 ]; then
                echo "│ ⚠️  CPU bottleneck detected (${cpu_util}%)                     │"
            elif [ "$cpu_util" -lt 20 ]; then
                echo "│ ✓ CPU idle - GPU bound (${cpu_util}%)                        │"
            else
                echo "│ ✓ Balanced CPU usage (${cpu_util}%)                         │"
            fi

            if echo "$swap_used" | grep -q "0B"; then
                echo "│ ✓ No swap usage - memory healthy                          │"
            else
                echo "│ ⚠️  Swap in use: $swap_used - memory pressure              │"
            fi
        fi
        echo "│                                                               │"
        echo "└───────────────────────────────────────────────────────────────┘"
    fi
}

if [ "$WATCH_MODE" = true ]; then
    while true; do
        get_performance_data
        echo ""
        echo "Refreshing in ${INTERVAL}s (Ctrl+C to stop)..."
        sleep "$INTERVAL"
    done
else
    get_performance_data
fi
