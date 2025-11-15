# DeepSeek-OCR API Service

Self-hosted OCR service using DeepSeek-OCR model, deployed on AWS EC2 with GPU support.

## Overview

This service exposes DeepSeek-OCR as a REST API for document text extraction and markdown conversion. It's optimized for AWS EC2 g4dn.xlarge instances (NVIDIA T4 GPU, 16GB GPU memory).

### Features

- ✅ **High-quality OCR**: Extract text from documents with 97% accuracy
- ✅ **Markdown conversion**: Convert documents to structured markdown
- ✅ **Multi-format support**: PDF, JPG, PNG, HEIC/HEIF
- ✅ **GPU-accelerated**: Fast inference with NVIDIA T4
- ✅ **REST API**: Easy integration with any application
- ✅ **Docker-based**: Reproducible deployments
- ✅ **Auto-scaling ready**: Designed for AWS infrastructure

## Architecture

```
┌─────────────┐      HTTP      ┌─────────────┐      GPU       ┌──────────────────┐
│   Client    │ ────────────> │    Nginx    │ ───────────> │ DeepSeek-OCR     │
│ Application │                │   (Port 80) │                │ FastAPI Service  │
└─────────────┘                └─────────────┘                │ (Port 8000)      │
                                                              │                  │
                                                              │ NVIDIA T4 GPU    │
                                                              │ 16GB VRAM        │
                                                              └──────────────────┘
```

## Quick Start

### Prerequisites

- AWS Account with EC2 access
- AWS CLI configured (`aws configure`)
- EC2 Key Pair created in your target region
- Basic knowledge of Linux and Docker

### 1. Deploy to AWS EC2

```bash
cd aws
chmod +x deploy-ec2.sh
./deploy-ec2.sh
```

Follow the prompts:
- Select AWS region (default: us-east-1)
- Enter your EC2 key pair name
- Optionally set an API key for authentication

**Deployment time**: ~10-15 minutes (includes model download ~6.6GB)

### 2. Verify Deployment

Once deployment completes, you'll receive:
- Instance ID
- Public IP address
- API endpoint

Test the service:

```bash
# Health check
curl http://<PUBLIC-IP>/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true,
  "gpu_name": "Tesla T4"
}
```

### 3. Process a Document

```bash
# Basic usage (default: gundam resolution)
curl -X POST "http://<PUBLIC-IP>/ocr/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"

# With specific resolution
curl -X POST "http://<PUBLIC-IP>/ocr/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf" \
  -F "resolution=large"

# Expected response:
{
  "success": true,
  "markdown_text": "# Extracted Document Content\n\n...",
  "error": null,
  "model_info": {
    "model_name": "deepseek-ai/DeepSeek-OCR",
    "device": "cuda",
    "max_tokens": 4096,
    "resolution": "gundam",
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": true
  }
}
```

## API Documentation

### Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true,
  "gpu_name": "Tesla T4"
}
```

#### `GET /resolutions`

Get available resolution options.

**Response:**
```json
{
  "resolutions": {
    "tiny": {
      "base_size": 512,
      "image_size": 512,
      "crop_mode": false,
      "description": "Fastest, lowest quality (512x512)"
    },
    "small": {
      "base_size": 640,
      "image_size": 640,
      "crop_mode": false,
      "description": "Fast, good for simple documents (640x640)"
    },
    "base": {
      "base_size": 1024,
      "image_size": 1024,
      "crop_mode": false,
      "description": "Balanced quality and speed (1024x1024)"
    },
    "large": {
      "base_size": 1280,
      "image_size": 1280,
      "crop_mode": false,
      "description": "High quality, slower (1280x1280)"
    },
    "gundam": {
      "base_size": 1024,
      "image_size": 640,
      "crop_mode": true,
      "description": "Recommended - Best quality with crop mode (1024/640)"
    }
  },
  "default": "gundam",
  "recommended": "gundam"
}
```

#### `POST /ocr/process`

Process a document and extract text as markdown.

**Parameters:**
- `file` (multipart/form-data): Document file (PDF, JPG, PNG, HEIC)
- `resolution` (optional, default: "gundam"): Resolution preset (tiny/small/base/large/gundam)
- `prompt` (optional): Custom extraction prompt
- `X-API-Key` (header, optional): API key if authentication enabled

**Resolution Options:**
- `tiny`: 512×512 - Fastest, lowest quality
- `small`: 640×640 - Fast, good for simple documents
- `base`: 1024×1024 - Balanced quality and speed
- `large`: 1280×1280 - High quality, slower
- `gundam`: 1024/640 crop mode - **Recommended**, best quality

**Response:**
```json
{
  "success": true,
  "markdown_text": "Extracted content in markdown format",
  "error": null,
  "model_info": {
    "model_name": "deepseek-ai/DeepSeek-OCR",
    "device": "cuda",
    "max_tokens": 4096,
    "resolution": "gundam",
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": true
  }
}
```

#### `POST /ocr/process-base64`

Process a base64-encoded image.

**Parameters:**
```json
{
  "image_base64": "base64_encoded_string",
  "file_extension": ".jpg",
  "resolution": "gundam",
  "prompt": "optional custom prompt"
}
```

#### `GET /docs`

Interactive API documentation (Swagger UI).

Access at: `http://<PUBLIC-IP>/docs`

## Configuration

### Environment Variables

Edit `.env` file on the EC2 instance:

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<PUBLIC-IP>
cd /opt/deepseek-ocr-service
nano .env
```

Available settings:

```bash
# Authentication
REQUIRE_API_KEY=true          # Enable/disable API key authentication
API_KEY=your-secret-key       # API key for authentication

# Model settings (advanced)
MODEL_NAME=deepseek-ai/DeepSeek-OCR
DEVICE=cuda
USE_BF16=true                 # Use bfloat16 precision
MAX_NEW_TOKENS=4096          # Maximum output tokens
```

After changes, restart the service:

```bash
cd /opt/deepseek-ocr-service
docker-compose restart
```

## Local Development

### Run Locally with Docker

```bash
# Build and run
docker-compose up --build

# Test
curl http://localhost:8000/health
```

### Run Locally without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Run service
python app.py
```

**Note**: Requires NVIDIA GPU with CUDA 11.8+ and 16GB+ VRAM.

## Cost Optimization

### Instance Costs (us-east-1)

| Instance Type | GPU | VRAM | Cost/Hour | Cost/Month (24/7) |
|--------------|-----|------|-----------|-------------------|
| g4dn.xlarge  | T4  | 16GB | $0.526    | ~$380             |
| g5.xlarge    | A10G| 24GB | $1.006    | ~$726             |

### Strategies to Reduce Costs

1. **Stop when not in use**:
   ```bash
   # Stop instance
   aws ec2 stop-instances --instance-ids <INSTANCE-ID>

   # Start when needed
   aws ec2 start-instances --instance-ids <INSTANCE-ID>
   ```

2. **Use Spot Instances**: Save up to 70%
   - Modify `deploy-ec2.sh` to request spot instances
   - Risk: Instance may be terminated if demand increases

3. **Schedule with Lambda**: Auto start/stop on schedule
   - Start: 9 AM weekdays
   - Stop: 6 PM weekdays
   - **Savings**: ~60% (only 45 hours/week vs 168 hours/week)

4. **Use Reserved Instances**: Save up to 40% for 1-year commitment

## Monitoring

### View Logs

```bash
# SSH into instance
ssh -i ~/.ssh/your-key.pem ubuntu@<PUBLIC-IP>

# View Docker logs
docker-compose -f /opt/deepseek-ocr-service/docker-compose.yml logs -f

# View system logs
tail -f /var/log/deepseek-ocr-setup.log
```

### CloudWatch Metrics

```bash
# Enable detailed monitoring (additional cost)
aws ec2 monitor-instances --instance-ids <INSTANCE-ID>
```

Monitor:
- CPU utilization
- GPU utilization (requires CloudWatch agent)
- Network traffic
- Disk I/O

## Troubleshooting

### Service not starting

```bash
# Check Docker status
docker ps -a

# Check logs
docker-compose logs

# Restart service
docker-compose down && docker-compose up -d
```

### Out of GPU memory

Reduce `MAX_NEW_TOKENS` in `.env`:

```bash
MAX_NEW_TOKENS=2048  # Default: 4096
```

### Slow inference

1. Verify GPU is being used:
   ```bash
   nvidia-smi
   ```

2. Check if using bfloat16:
   ```bash
   grep USE_BF16 .env  # Should be: USE_BF16=true
   ```

### Connection timeout

1. Check security group allows port 80:
   ```bash
   aws ec2 describe-security-groups --group-ids <SG-ID>
   ```

2. Check nginx is running:
   ```bash
   docker ps | grep nginx
   ```

## Integration Example

### Python Client

```python
import requests

API_URL = "http://<PUBLIC-IP>/ocr/process"
API_KEY = "your-secret-key"  # If authentication enabled

def extract_text(file_path: str, resolution: str = "gundam") -> str:
    """
    Extract text from document

    Args:
        file_path: Path to document file
        resolution: Resolution preset (tiny/small/base/large/gundam)
    """

    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'resolution': resolution}
        headers = {'X-API-Key': API_KEY}  # If authentication enabled

        response = requests.post(
            API_URL,
            files=files,
            data=data,
            headers=headers
        )
        response.raise_for_status()

        result = response.json()

        if result['success']:
            return result['markdown_text']
        else:
            raise Exception(result['error'])

# Usage
text = extract_text('/path/to/document.pdf')  # Uses gundam (recommended)
print(text)

# Or with specific resolution
text = extract_text('/path/to/simple-document.pdf', resolution='small')  # Faster
print(text)
```

### Node.js Client

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function extractText(filePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));

    const response = await axios.post(
        'http://<PUBLIC-IP>/ocr/process',
        form,
        {
            headers: {
                ...form.getHeaders(),
                'X-API-Key': 'your-secret-key'  // If authentication enabled
            }
        }
    );

    if (response.data.success) {
        return response.data.markdown_text;
    } else {
        throw new Error(response.data.error);
    }
}

// Usage
extractText('/path/to/document.pdf')
    .then(text => console.log(text))
    .catch(err => console.error(err));
```

## Security Best Practices

1. **Enable API Key Authentication**:
   ```bash
   REQUIRE_API_KEY=true
   API_KEY=<generate-strong-random-key>
   ```

2. **Restrict Security Group**:
   - Only allow your application's IP addresses
   - Don't use `0.0.0.0/0` in production

3. **Use HTTPS**:
   - Add SSL certificate to nginx
   - Use AWS Certificate Manager + Load Balancer

4. **Regular Updates**:
   ```bash
   apt-get update && apt-get upgrade -y
   docker-compose pull && docker-compose up -d
   ```

## Scaling for Production

### Option 1: Horizontal Scaling (Multiple Instances)

```
Load Balancer
    ├── DeepSeek-OCR Instance 1
    ├── DeepSeek-OCR Instance 2
    └── DeepSeek-OCR Instance 3
```

### Option 2: AWS Batch

For batch processing large volumes of documents.

### Option 3: SageMaker

Deploy as a SageMaker endpoint for managed scaling.

## Support

### Common Issues

- **Model not loading**: Check GPU memory (need 16GB+ VRAM)
- **Slow startup**: Model download takes ~5 minutes first time
- **OOM errors**: Reduce MAX_NEW_TOKENS or batch size

### Resources

- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## License

DeepSeek-OCR model is available under MIT License.
This deployment code is also MIT Licensed.

## Changelog

### v1.0.0 (2025-01-16)
- Initial release
- Support for PDF, JPG, PNG, HEIC
- AWS EC2 deployment automation
- Docker-based deployment
- REST API with FastAPI
- GPU acceleration with NVIDIA T4
