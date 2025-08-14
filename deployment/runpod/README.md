# DotsOCR RunPod Deployment

This directory contains the deployment configuration for running dots.ocr on RunPod using vLLM.

## Files

- `Dockerfile.vllm-runpod` - Docker image configuration for RunPod deployment
- `verify_model.py` - Model verification script
- `test_endpoint.py` - Endpoint testing script
- `README.md` - This file

## Quick Start

### 1. Build Docker Image Locally (Optional)

```bash
# From the deployment/runpod directory
docker build -t dots-ocr-runpod -f Dockerfile.vllm-runpod .
```

### 2. Deploy on RunPod

1. **Use GitHub Container Registry Image** (Recommended):
   - Image: `ghcr.io/[your-username]/winit-dots-ocr/dots-ocr-runpod:latest`
   - The GitHub Action will automatically build and push the image

2. **Create RunPod Template**:
   - Go to RunPod Console
   - Create new template
   - Container Image: Use the GitHub Container Registry image above
   - Container Disk: At least 50GB (model is large)
   - Expose HTTP Ports: `8000`
   - Environment Variables: (Optional overrides)
     - `MAX_MODEL_LEN=8192`
     - `GPU_MEMORY_UTILIZATION=0.95`

3. **Hardware Requirements**:
   - Minimum: RTX 4090 (24GB VRAM)
   - Recommended: A100 40GB or H100
   - Memory: At least 32GB RAM

### 3. Test Deployment

```bash
# Install dependencies
pip install requests

# Test the endpoint
python test_endpoint.py https://your-runpod-endpoint-url

# Test with custom image
python test_endpoint.py https://your-runpod-endpoint-url --image path/to/test/image.jpg

# Test with API key (if configured)
python test_endpoint.py https://your-runpod-endpoint-url --api-key your-api-key
```

## Model Information

- **Model**: `rednote-hilab/dots.ocr`
- **Architecture**: Vision-Language Model for OCR
- **Size**: ~7B parameters
- **Storage Path**: `/models/DotsOCR` (no dots in directory name for Python imports)

## API Endpoints

The deployment provides OpenAI-compatible API endpoints:

- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion with OCR capability

### Example Usage

```python
import requests
import base64

# Encode image
with open("document.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# OCR Request
response = requests.post("https://your-endpoint/v1/chat/completions", json={
    "model": "rednote-hilab/dots.ocr",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    }],
    "max_tokens": 1000,
    "temperature": 0.1
})

result = response.json()
extracted_text = result["choices"][0]["message"]["content"]
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce `GPU_MEMORY_UTILIZATION` to 0.85 or 0.8
   - Use smaller `MAX_MODEL_LEN`
   - Ensure sufficient GPU VRAM

2. **Slow Loading**:
   - First request takes longer (model loading)
   - Consider using persistent storage for model cache

3. **Model Not Found**:
   - Check if model downloaded correctly during build
   - Run verification: `python verify_model.py`

### Debug Commands

```bash
# Check model files inside container
docker run -it dots-ocr-runpod ls -la /models/DotsOCR

# Verify model installation
docker run -it dots-ocr-runpod python /app/verify_model.py

# Check vLLM logs
docker logs <container_id>
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `rednote-hilab/dots.ocr` | HuggingFace model identifier |
| `MODEL_PATH` | `/models/DotsOCR` | Model storage path |
| `MAX_MODEL_LEN` | `8192` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU memory usage ratio |
| `MAX_NUM_SEQS` | `256` | Maximum concurrent sequences |
| `TENSOR_PARALLEL_SIZE` | `1` | Tensor parallelism degree |
| `TRUST_REMOTE_CODE` | `true` | Allow remote code execution |

## GitHub Actions

The repository includes a GitHub Action (`.github/workflows/build-runpod-docker.yml`) that automatically:

1. Builds the Docker image on push to main or runpod-deployment branches
2. Pushes to GitHub Container Registry
3. Validates the Docker configuration
4. Provides deployment instructions

To trigger a build:
- Push changes to deployment files
- Create a pull request
- Use "Repository → Actions → Build and Push RunPod Docker Image → Run workflow"

## Security Notes

- The model requires `TRUST_REMOTE_CODE=true` for proper functionality
- Images are stored in GitHub Container Registry with repository access controls
- API endpoints should be secured in production environments
- Consider implementing rate limiting and authentication