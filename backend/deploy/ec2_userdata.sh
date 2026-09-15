#!/bin/bash
set -e

# ========================================
# BHUVISION GeoChat-7B EC2 Server Setup
# Runs on first boot via user-data
# ========================================

exec > /var/log/geochat-setup.log 2>&1
echo "=== GeoChat-7B Setup Started at $(date) ==="

# Install system dependencies
apt-get update -y
apt-get install -y python3-pip nginx

# Create app directory
mkdir -p /opt/geochat
cd /opt/geochat

# Create the FastAPI server
cat > server.py << 'SERVEREOF'
"""GeoChat-7B Inference Server for BHUVISION.

Exposes a simple REST API for VQA, captioning, and grounded detection
on remote sensing / satellite imagery.
"""

import base64
import io
import json
import os
import re
import gc
import logging

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("geochat-server")

app = FastAPI(title="GeoChat-7B API", version="1.0.0")

# ── Global model state ──────────────────────────────────────────────
MODEL = None
TOKENIZER = None
IMAGE_PROCESSOR = None
CONV_MODE = "llava_v1"


class InferenceRequest(BaseModel):
    image: str  # base64-encoded image
    prompt: str = "Describe this image."
    task: str = "vqa"  # vqa | caption | detect
    max_tokens: int = 512
    temperature: float = 0.2


class InferenceResponse(BaseModel):
    generated_text: str
    model: str = "MBZUAI/geochat-7B"


def load_model():
    """Load GeoChat-7B model, tokenizer, and image processor."""
    global MODEL, TOKENIZER, IMAGE_PROCESSOR
    if MODEL is not None:
        return

    logger.info("Loading GeoChat-7B model...")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        CLIPImageProcessor,
    )

    model_id = "MBZUAI/geochat-7B"
    hf_token = os.environ.get("HF_TOKEN", None)

    TOKENIZER = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        use_fast=False,
    )

    MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    MODEL.eval()

    # CLIP image processor for the vision encoder
    IMAGE_PROCESSOR = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14-336"
    )

    logger.info("GeoChat-7B loaded successfully!")
    logger.info(f"Model device: {MODEL.device}")
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        used = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"GPU memory: {used:.1f} / {mem:.1f} GB")


def prepare_image(image_b64: str):
    """Decode base64 image and preprocess for CLIP."""
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image_tensor = IMAGE_PROCESSOR.preprocess(image, return_tensors="pt")["pixel_values"]
    return image_tensor.half().cuda()


def build_prompt(task: str, user_prompt: str) -> str:
    """Build the prompt for GeoChat based on task type."""
    if task == "caption":
        return "Describe this satellite image in detail."
    elif task == "detect":
        target = user_prompt.strip() or "objects"
        return f"Identify and locate all {target} in this image. For each object, provide its location."
    else:  # vqa
        return user_prompt


DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    """Tokenize prompt with image token handling (LLaVA style)."""
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]
    input_ids = []
    for i, chunk in enumerate(prompt_chunks):
        if i > 0:
            input_ids.append(image_token_index)
        # Remove BOS token from subsequent chunks
        if i > 0 and len(chunk) > 0 and chunk[0] == tokenizer.bos_token_id:
            chunk = chunk[1:]
        input_ids.extend(chunk)
    return torch.tensor(input_ids, dtype=torch.long)


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "loading",
        "model": "MBZUAI/geochat-7B",
        "gpu_available": torch.cuda.is_available(),
    }


@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded yet")

    try:
        # Prepare image
        image_tensor = prepare_image(req.image)

        # Build the conversation prompt
        task_prompt = build_prompt(req.task, req.prompt)
        full_prompt = (
            f"A chat between a curious human and an artificial intelligence assistant. "
            f"The assistant gives helpful, detailed, and polite answers to the human's questions. "
            f"USER: {DEFAULT_IMAGE_TOKEN}\n{task_prompt} ASSISTANT:"
        )

        # Tokenize
        input_ids = tokenizer_image_token(full_prompt, TOKENIZER).unsqueeze(0).cuda()

        # Generate
        with torch.inference_mode():
            output_ids = MODEL.generate(
                input_ids,
                images=image_tensor,
                do_sample=req.temperature > 0,
                temperature=max(req.temperature, 0.01),
                max_new_tokens=req.max_tokens,
                use_cache=True,
            )

        # Decode only the generated tokens (skip the input)
        generated = TOKENIZER.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return InferenceResponse(generated_text=generated)

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
SERVEREOF

# Create requirements file
cat > requirements.txt << 'REQEOF'
fastapi==0.115.0
uvicorn[standard]==0.30.0
transformers>=4.37.0
accelerate>=0.26.0
torch>=2.1.0
Pillow>=10.0.0
sentencepiece
protobuf
REQEOF

# Create systemd service
cat > /etc/systemd/system/geochat.service << 'SVCEOF'
[Unit]
Description=GeoChat-7B Inference Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/geochat
Environment="HF_TOKEN=__HF_TOKEN_PLACEHOLDER__"
Environment="TRANSFORMERS_CACHE=/opt/geochat/cache"
Environment="HF_HOME=/opt/geochat/cache"
ExecStart=/usr/bin/python3 server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCEOF

# Install Python dependencies
pip3 install -r requirements.txt

# Pre-download the model
echo "=== Pre-downloading GeoChat-7B model ==="
python3 -c "
import os
os.environ['HF_HOME'] = '/opt/geochat/cache'
os.environ['TRANSFORMERS_CACHE'] = '/opt/geochat/cache'
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPImageProcessor
token = os.environ.get('HF_TOKEN', '__HF_TOKEN_PLACEHOLDER__')
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('MBZUAI/geochat-7B', trust_remote_code=True, token=token, use_fast=False)
print('Downloading CLIP processor...')
CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
print('Downloading model weights (~14GB)...')
AutoModelForCausalLM.from_pretrained('MBZUAI/geochat-7B', trust_remote_code=True, token=token, torch_dtype='auto', low_cpu_mem_usage=True)
print('Download complete!')
"

# Enable and start the service
systemctl daemon-reload
systemctl enable geochat
systemctl start geochat

echo "=== GeoChat-7B Setup Complete at $(date) ==="
echo "Server running on port 8080"
