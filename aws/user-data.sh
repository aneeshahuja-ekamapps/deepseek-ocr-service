#!/bin/bash

###############################################################################
# EC2 User Data Script for DeepSeek-OCR Service
# Automatically runs on instance first boot
###############################################################################

set -e

LOG_FILE="/var/log/deepseek-ocr-setup.log"
exec > >(tee -a $LOG_FILE) 2>&1

echo "=========================================="
echo "DeepSeek-OCR Service Setup Starting"
echo "Time: $(date)"
echo "=========================================="

# Update system
echo "[1/8] Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install Docker
echo "[2/8] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu
    systemctl enable docker
    systemctl start docker
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# Install Docker Compose
echo "[3/8] Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed"
else
    echo "✅ Docker Compose already installed"
fi

# Install NVIDIA Container Toolkit
echo "[4/8] Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-container-toolkit &> /dev/null; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

    apt-get update
    apt-get install -y nvidia-container-toolkit
    systemctl restart docker
    echo "✅ NVIDIA Container Toolkit installed"
else
    echo "✅ NVIDIA Container Toolkit already installed"
fi

# Verify GPU access
echo "[5/8] Verifying GPU access..."
nvidia-smi
if [ $? -eq 0 ]; then
    echo "✅ GPU detected successfully"
else
    echo "❌ GPU not detected!"
fi

# Create application directory
echo "[6/8] Setting up application directory..."
APP_DIR="/opt/deepseek-ocr-service"
mkdir -p $APP_DIR
cd $APP_DIR

# Download service files from repository or create them
echo "[7/8] Creating service files..."

# Create app.py
cat > app.py << 'EOF'
"""
DeepSeek-OCR FastAPI Service
Exposes OCR processing via REST API
"""

import os
import io
import base64
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import DeepSeek-OCR dependencies
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image

from config import settings


# Global model storage
model_instance = None
tokenizer_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model_instance, tokenizer_instance

    print("🚀 Loading DeepSeek-OCR model...")
    print(f"   Model: {settings.MODEL_NAME}")
    print(f"   Device: {settings.DEVICE}")

    try:
        # Load tokenizer
        tokenizer_instance = AutoTokenizer.from_pretrained(
            settings.MODEL_NAME,
            trust_remote_code=True
        )

        # Load model
        model_instance = AutoModel.from_pretrained(
            settings.MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if settings.USE_BF16 else torch.float16
        ).to(settings.DEVICE)

        model_instance.eval()

        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        raise

    yield

    # Cleanup
    print("🔄 Shutting down, cleaning up resources...")
    del model_instance
    del tokenizer_instance
    torch.cuda.empty_cache()


# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek-OCR API Service",
    description="Self-hosted OCR service using DeepSeek-OCR model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class OCRResponse(BaseModel):
    success: bool
    markdown_text: str
    error: Optional[str] = None
    model_info: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_available: bool
    gpu_name: Optional[str] = None


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """Verify API key if authentication is enabled"""
    if not settings.REQUIRE_API_KEY:
        return True

    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")

    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True


def process_ocr(image_file: str, prompt: str = None) -> str:
    """
    Process OCR using DeepSeek-OCR model

    Args:
        image_file: Path to image file
        prompt: Optional custom prompt (default: markdown conversion)

    Returns:
        Extracted markdown text
    """
    global model_instance, tokenizer_instance

    if model_instance is None or tokenizer_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Default prompt for document to markdown conversion
        if prompt is None:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."

        # Run inference
        with torch.no_grad():
            result = model_instance.infer(
                tokenizer=tokenizer_instance,
                prompt=prompt,
                image_file=image_file,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True
            )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return HealthResponse(
        status="healthy" if model_instance is not None else "unhealthy",
        model_loaded=model_instance is not None,
        device=settings.DEVICE,
        gpu_available=gpu_available,
        gpu_name=gpu_name
    )


@app.post("/ocr/process", response_model=OCRResponse)
async def process_document(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
    x_api_key: Optional[str] = Header(None)
):
    """
    Process document and extract text as markdown

    Args:
        file: Image/PDF file to process
        prompt: Optional custom prompt for extraction
        x_api_key: API key for authentication (if enabled)

    Returns:
        Extracted markdown text
    """
    # Verify API key
    verify_api_key(x_api_key)

    # Validate file type
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.pdf', '.heic', '.heif']
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        # Read file content
        content = await file.read()

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Process OCR
            markdown_text = process_ocr(temp_path, prompt)

            return OCRResponse(
                success=True,
                markdown_text=markdown_text,
                error=None,
                model_info={
                    "model_name": settings.MODEL_NAME,
                    "device": settings.DEVICE,
                    "max_tokens": settings.MAX_NEW_TOKENS
                }
            )

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        return OCRResponse(
            success=False,
            markdown_text="",
            error=str(e),
            model_info={}
        )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=1,
        log_level=settings.LOG_LEVEL.lower()
    )
EOF

# Create config.py
cat > config.py << 'EOF'
"""
Configuration for DeepSeek-OCR API Service
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Model configuration
    MODEL_NAME: str = "deepseek-ai/DeepSeek-OCR"
    DEVICE: str = "cuda"
    USE_BF16: bool = True
    MAX_NEW_TOKENS: int = 4096

    # API configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Security
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
    API_KEY: str = os.getenv("API_KEY", "your-secret-api-key-change-this")

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
pydantic==2.9.2
pydantic-settings==2.6.0
torch==2.6.0
transformers==4.49.0
accelerate==1.2.0
pillow==11.0.0
flash-attn==2.7.3
python-dotenv==1.0.1
EOF

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    curl \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation || echo "Flash attention installation failed, continuing..."

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY config.py .

RUN mkdir -p /root/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "app.py"]
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  deepseek-ocr:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: deepseek-ocr-service
    restart: unless-stopped

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    environment:
      - REQUIRE_API_KEY=${REQUIRE_API_KEY:-false}
      - API_KEY=${API_KEY:-your-secret-api-key}
      - CUDA_VISIBLE_DEVICES=0

    ports:
      - "8000:8000"

    volumes:
      - huggingface-cache:/root/.cache/huggingface
      - ./logs:/app/logs

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

volumes:
  huggingface-cache:
    driver: local
EOF

# Create .env file
cat > .env << 'EOF'
REQUIRE_API_KEY=false
API_KEY=your-secret-api-key-change-this
EOF

echo "✅ Service files created"

# Build and start Docker containers
echo "[8/8] Building and starting Docker containers..."
echo "⚠️  This will download ~6.6GB model weights and build the image"
echo "⚠️  Expected time: 10-15 minutes"

docker-compose up -d --build

echo ""
echo "=========================================="
echo "✅ DeepSeek-OCR Service Setup Complete!"
echo "=========================================="
echo ""
echo "Checking service status in 30 seconds..."
sleep 30

docker-compose logs --tail=50

echo ""
echo "Service should be accessible at:"
echo "  http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/health"
echo ""
echo "Monitor logs with:"
echo "  docker-compose logs -f"
echo ""
