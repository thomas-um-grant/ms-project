# Device-Agnostic ColPali Indexer

This refactored ColPali implementation provides a device-agnostic system that automatically detects and configures the optimal hardware setup for your machine (CUDA, MPS, or CPU).

## Key Features

- **Automatic Device Detection**: Automatically chooses the best available device (CUDA > MPS > CPU)
- **Device Override**: Force specific device usage when needed
- **Numerical Stability**: Built-in NaN handling and dtype management
- **Cross-Platform**: Works on NVIDIA GPUs, Apple Silicon, and CPU-only systems

## Usage

### Basic Usage (Auto-detect)

```python
from test_colpali import MultiModalRAG, DeviceConfig
from pathlib import Path
from PIL import Image

# Auto-detect the best device configuration
indexer = MultiModalRAG(
    data_dir=Path("./rag_store"),
    batch_size=8,
)

# Index your images
images = [Image.open("image1.png"), Image.open("image2.png")]
metadata = [
    {"doc_id": "doc1", "page": 0, "text": "Content of page 1"},
    {"doc_id": "doc1", "page": 1, "text": "Content of page 2"},
]

indexer.index_image_pages(images, metadata)

# Perform retrieval
results = indexer.retrieve("your query here", top_k=5)
for metadata, score in results:
    print(f"Document: {metadata['doc_id']}, Score: {score}")
```

### Force Specific Device

```python
# Force CPU usage (most stable)
device_config = DeviceConfig.auto_detect(preferred_device="cpu")

# Force MPS usage (Apple Silicon)
device_config = DeviceConfig.auto_detect(preferred_device="mps")

# Force CUDA usage (NVIDIA GPUs)
device_config = DeviceConfig.auto_detect(preferred_device="cuda")

indexer = MultiModalRAG(
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
    batch_size=4,  # Reduced from default 8
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

### MultiModalRAG

- `__init__(data_dir, model_name, batch_size, device_config)`: Initialize indexer
- `index_image_pages(image_paths, metadata_list)`: Index images with metadata
- `retrieve(query_text, top_k)`: Retrieve similar documents

## Error Handling

The system includes comprehensive error handling:

- **NaN Detection**: Automatically detects and replaces NaN values
- **Device Fallback**: Falls back to more stable configurations
- **Memory Management**: Safe tensor conversions between devices
- **File Management**: Proper cleanup and error messages
