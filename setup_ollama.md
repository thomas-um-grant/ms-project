# Setup Ollama with Qwen2-VL for Faster Inference

## Installation

1. **Install Ollama** (if not already installed):

   ```bash
   # On macOS
   brew install ollama

   # Or download from https://ollama.ai/download
   ```

2. **Start Ollama server**:

   ```bash
   ollama serve
   ```

3. **Pull the Qwen2.5-VL model**:
   ```bash
   ollama pull qwen2.5vl:7b
   ```

## Usage

To use the Ollama-based generation model instead of the direct HuggingFace model, simply change the model name in your configuration:

```python
# Instead of:
model = setup_generation_model("colqwen2_gen", device_config)

# Use:
model = setup_generation_model("colqwen2_ollama", device_config)
```

## Benefits

- **Faster inference** on ARM-based Macs (M1/M2/M3)
- **Lower memory usage** - Ollama optimizes model loading and memory management
- **Better performance** - Ollama uses optimized inference engines
- **Same interface** - Drop-in replacement for the existing model

## Verification

Test that Ollama is working correctly:

```bash
# Test basic chat
ollama run qwen2.5vl:7b "Hello, how are you?"

# Test with image (you'll need to provide a local image path)
ollama run qwen2.5vl:7b "What's in this image?" --image /path/to/your/image.jpg
```

## Troubleshooting

- **Connection issues**: Make sure Ollama server is running (`ollama serve`)
- **Model not found**: Ensure you've pulled the model (`ollama pull qwen2.5vl:7b`)
- **Port conflicts**: Ollama uses port 11434 by default
- **Memory issues**: Ollama will automatically manage GPU/CPU memory allocation

## Configuration

You can customize Ollama settings by modifying the `ColQwen2OllamaModel` parameters:

- `ollama_url`: Change if Ollama is running on a different host/port
- `model_name`: Use a different Qwen2.5-VL variant if available
- `temperature`: Adjust response creativity (default: 0.8)
- `num_predict`: Adjust maximum response length (default: 512 tokens)
