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

# Resolution configurations for DeepSeek-OCR
# Based on: https://huggingface.co/spaces/khang119966/DeepSeek-OCR-DEMO
SIZE_CONFIGS = {
    "tiny": {
        "base_size": 512,
        "image_size": 512,
        "crop_mode": False,
        "description": "Fastest, lowest quality (512x512)"
    },
    "small": {
        "base_size": 640,
        "image_size": 640,
        "crop_mode": False,
        "description": "Fast, good for simple documents (640x640)"
    },
    "base": {
        "base_size": 1024,
        "image_size": 1024,
        "crop_mode": False,
        "description": "Balanced quality and speed (1024x1024)"
    },
    "large": {
        "base_size": 1280,
        "image_size": 1280,
        "crop_mode": False,
        "description": "High quality, slower (1280x1280)"
    },
    "gundam": {
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
        "description": "Recommended - Best quality with crop mode (1024/640)"
    }
}


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


def process_ocr(
    image_file: str,
    prompt: str = None,
    resolution: str = "gundam"
) -> str:
    """
    Process OCR using DeepSeek-OCR model

    Args:
        image_file: Path to image file
        prompt: Optional custom prompt (default: markdown conversion)
        resolution: Resolution preset (tiny/small/base/large/gundam)

    Returns:
        Extracted markdown text
    """
    global model_instance, tokenizer_instance

    if model_instance is None or tokenizer_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get resolution configuration
    resolution = resolution.lower()
    if resolution not in SIZE_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resolution. Choose from: {', '.join(SIZE_CONFIGS.keys())}"
        )

    config = SIZE_CONFIGS[resolution]

    try:
        # Default prompt for document to markdown conversion
        if prompt is None:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."

        # Run inference with resolution settings
        with torch.no_grad():
            result = model_instance.infer(
                tokenizer=tokenizer_instance,
                prompt=prompt,
                image_file=image_file,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"]
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


@app.get("/resolutions")
async def get_resolutions():
    """
    Get available resolution options

    Returns:
        Dictionary of available resolutions with their configurations
    """
    return {
        "resolutions": SIZE_CONFIGS,
        "default": "gundam",
        "recommended": "gundam"
    }


@app.post("/ocr/process", response_model=OCRResponse)
async def process_document(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
    resolution: Optional[str] = "gundam",
    x_api_key: Optional[str] = Header(None)
):
    """
    Process document and extract text as markdown

    Args:
        file: Image/PDF file to process
        prompt: Optional custom prompt for extraction
        resolution: Resolution preset (tiny/small/base/large/gundam). Default: gundam
        x_api_key: API key for authentication (if enabled)

    Returns:
        Extracted markdown text

    Resolution Options:
        - tiny: 512x512 - Fastest, lowest quality
        - small: 640x640 - Fast, good for simple documents
        - base: 1024x1024 - Balanced quality and speed
        - large: 1280x1280 - High quality, slower
        - gundam: 1024/640 crop mode - Recommended, best quality
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
            # Process OCR with resolution
            markdown_text = process_ocr(temp_path, prompt, resolution)

            # Get resolution config for response
            config = SIZE_CONFIGS[resolution.lower()]

            return OCRResponse(
                success=True,
                markdown_text=markdown_text,
                error=None,
                model_info={
                    "model_name": settings.MODEL_NAME,
                    "device": settings.DEVICE,
                    "max_tokens": settings.MAX_NEW_TOKENS,
                    "resolution": resolution,
                    "base_size": config["base_size"],
                    "image_size": config["image_size"],
                    "crop_mode": config["crop_mode"]
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


@app.post("/ocr/process-base64", response_model=OCRResponse)
async def process_document_base64(
    image_base64: str,
    file_extension: str = ".jpg",
    prompt: Optional[str] = None,
    resolution: Optional[str] = "gundam",
    x_api_key: Optional[str] = Header(None)
):
    """
    Process document from base64-encoded image

    Args:
        image_base64: Base64-encoded image data
        file_extension: File extension (.jpg, .png, etc.)
        prompt: Optional custom prompt
        resolution: Resolution preset (tiny/small/base/large/gundam). Default: gundam
        x_api_key: API key for authentication

    Returns:
        Extracted markdown text
    """
    # Verify API key
    verify_api_key(x_api_key)

    try:
        # Decode base64
        image_data = base64.b64decode(image_base64)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name

        try:
            # Process OCR with resolution
            markdown_text = process_ocr(temp_path, prompt, resolution)

            # Get resolution config for response
            config = SIZE_CONFIGS[resolution.lower()]

            return OCRResponse(
                success=True,
                markdown_text=markdown_text,
                error=None,
                model_info={
                    "model_name": settings.MODEL_NAME,
                    "device": settings.DEVICE,
                    "resolution": resolution,
                    "base_size": config["base_size"],
                    "image_size": config["image_size"],
                    "crop_mode": config["crop_mode"]
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
        workers=1,  # Single worker for GPU (multiple workers share GPU)
        log_level=settings.LOG_LEVEL.lower()
    )
