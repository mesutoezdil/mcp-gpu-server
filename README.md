mcp-name: io.github.mesutoezdil/mcp-gpu-server

# mcp-gpu-server

MCP server for GPU monitoring. Queries NVIDIA GPUs via NVML or nvidia-smi.

## Tools

    gpu_info         name, driver, CUDA version
    gpu_utilization  core and memory utilization %
    gpu_vram         total / used / free VRAM in MiB
    gpu_temperature  temperature in C
    gpu_stats        full snapshot of all metrics

## Install

    pip install mcp-gpu-server

## MCP config

    {
      "mcpServers": {
        "gpu": {
          "command": "mcp-gpu-server"
        }
      }
    }

## Requirements

Python 3.10 or higher with NVIDIA GPU and drivers installed.
