# Device-Agnostic models

These models provide a device-agnostic system that automatically detects and configures the optimal hardware setup for the machine (CUDA, MPS, or CPU), unless the device has been specified in the configurations.

## Key Features

- **Automatic Device Detection**: Automatically chooses the best available device (CUDA > MPS > CPU)
- **Device Override**: Force specific device usage when needed
- **Numerical Stability**: Built-in NaN handling and dtype management
- **Cross-Platform**: Works on NVIDIA GPUs, Apple Silicon, and CPU-only systems

## Usage

### Force Specific Device

```python
# Force CPU usage (most stable)
device_config = DeviceConfig.auto_detect(preferred_device="cpu")

# Force MPS usage (Apple Silicon)
device_config = DeviceConfig.auto_detect(preferred_device="mps")

# Force CUDA usage (NVIDIA GPUs)
device_config = DeviceConfig.auto_detect(preferred_device="cuda")

# Example with multimodal RAG
embedder = MultiModalRAG(
    data_dir=Path("./rag_store"),
    device_config=device_config,
)
```

### Custom Configuration

```python
from dataclasses import dataclass
import torch

# Create custom device configuration
custom_config = DeviceConfig(
    device="cuda",
    dtype=torch.float16,
    device_map="auto"
)

indexer = MultiModalRAG(
    data_dir=Path("./rag_store"),
    device_config=custom_config,
)
```

## Device Configurations

| Device | Default dtype | Device Map | Best for         |
| ------ | ------------- | ---------- | ---------------- |
| CUDA   | float16       | auto       | NVIDIA GPUs      |
| MPS    | float16       | mps        | Apple Silicon    |
| CPU    | float32       | cpu        | CPU-only systems |

## Troubleshooting

### MPS Issues

If you encounter NaN values or instability with MPS, the system will automatically fallback to CPU. You can also force CPU usage:

```python
device_config = DeviceConfig.auto_detect(preferred_device="cpu")
```

### Memory Issues

Reduce batch size if you encounter out-of-memory errors:

```python
indexer = MultiModalRAG(
    data_dir=Path("./rag_store"),
    batch_size=4,  # Reduced from default
)
```

### Performance Optimization

- **CUDA**: Use float16 for fastest performance
- **MPS**: Use float16 but expect occasional fallbacks
- **CPU**: Use float32 for stability

## Class Structure

### DeviceConfig

- `auto_detect(preferred_device=None)`: Auto-detect optimal configuration
- `device`: Target device ("cuda", "mps", "cpu")
- `dtype`: Tensor data type (torch.float16/torch.float32)
- `device_map`: Device mapping strategy

## Error Handling

The system includes comprehensive error handling:

- **NaN Detection**: Automatically detects and replaces NaN values
- **Device Fallback**: Falls back to more stable configurations
- **Memory Management**: Safe tensor conversions between devices
- **File Management**: Proper cleanup and error messages
