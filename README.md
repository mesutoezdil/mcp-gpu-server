# mcp-gpu-server

mcp-name: io.github.mesutoezdil/mcp-gpu-server

MCP server for GPU monitoring. Queries NVIDIA GPUs via NVML (preferred) or `nvidia-smi` fallback.

## Tools

| Tool | Returns |
|------|---------|
| `gpu_info` | Name, driver, CUDA version |
| `gpu_utilization` | Core and memory utilization % |
| `gpu_vram` | Total / used / free VRAM (MiB) |
| `gpu_temperature` | Temperature in °C |
| `gpu_stats` | Full snapshot of all metrics |

## Install

```bash
pip install mcp-gpu-server
```

## MCP config

```json
{
  "mcpServers": {
    "gpu": {
      "command": "mcp-gpu-server"
    }
  }
}
```

## Requirements

- Python 3.10+
- NVIDIA GPU with drivers installed
