"""Deploy GeoChat-7B on SageMaker with custom inference code and architecture.

Uses the HuggingFace PyTorch inference container with custom inference.py
and the complete GeoChat model architecture package (configuration, modeling,
processor, and engine).
"""

import argparse
import base64
import io
import json
import logging
import os
import tarfile
import time

import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ROLE_ARN = "arn:aws:iam::347694270791:role/BHUVISION-SageMakerExecutionRole"
ENDPOINT_NAME = os.getenv("BHUVISION_GEOCHAT_ENDPOINT_NAME", "bhuvision-geochat-7b")

# ── Inference code that runs inside SageMaker ─────────────────────
INFERENCE_PY = r'''
import base64
import io
import json
import logging
import os
import subprocess
import sys
import time

import torch
from PIL import Image

# Add current code directory to sys.path
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Diagnostic: log CUDA/GPU environment at import time ──────────────
logger.info("=" * 60)
logger.info("[DIAG] PyTorch version: %s", torch.__version__)
logger.info("[DIAG] CUDA available: %s", torch.cuda.is_available())
logger.info("[DIAG] CUDA version (compiled): %s", torch.version.cuda)
try:
    nvidia_smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
    logger.info("[DIAG] nvidia-smi output:\n%s", nvidia_smi[:500])
except Exception as e:
    logger.warning("[DIAG] nvidia-smi not available: %s", e)
if torch.cuda.is_available():
    logger.info("[DIAG] GPU count: %s", torch.cuda.device_count())
    logger.info("[DIAG] GPU 0 name: %s", torch.cuda.get_device_name(0))
logger.info("=" * 60)

# Import GeoChat custom architecture
from geochat.src.configuration_geochat import GeoChatConfig
from geochat.src.modeling_geochat import GeoChatLlamaForCausalLM
from geochat.src.processing_geochat import GeoChatProcessor
from geochat.inference.sar import load_image_input, sar1_to_rgb

# Register with transformers Auto classes
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("geochat", GeoChatConfig)
AutoModelForCausalLM.register(GeoChatConfig, GeoChatLlamaForCausalLM)


# ── Lazy-loading wrapper ──────────────────────────────────────────────
class LazyGeoChatEngine:
    """Delays actual model loading until the first prediction request.

    This avoids the problem where torch.cuda.is_available() returns False
    during the very early SageMaker worker fork/init, but becomes True
    once the CUDA driver is fully initialised.
    """

    def __init__(self, model_id):
        self.model_id = model_id
        self._engine = None
        self._load_error = None

    def _ensure_loaded(self):
        if self._engine is not None:
            return
        if self._load_error is not None:
            raise self._load_error

        logger.info("[LazyEngine] First request – loading model now...")
        logger.info("[LazyEngine] CUDA available: %s", torch.cuda.is_available())

        try:
            use_gpu = torch.cuda.is_available()
            device = "cuda" if use_gpu else "cpu"
            dtype = torch.float16 if use_gpu else torch.float32

            logger.info("[LazyEngine] Loading config from %s ...", self.model_id)
            config = GeoChatConfig.from_pretrained(self.model_id)

            logger.info("[LazyEngine] Loading model weights (%s, %s)...", device, dtype)
            model = GeoChatLlamaForCausalLM.from_pretrained(
                self.model_id,
                config=config,
                torch_dtype=dtype,
                device_map="auto" if use_gpu else None,
                low_cpu_mem_usage=True,
            )
            model.eval()

            logger.info("[LazyEngine] Loading processor...")
            processor = GeoChatProcessor(
                model_name_or_path=self.model_id,
                vision_model_name=config.mm_vision_tower,
            )

            # Store as a simple namespace
            class _Engine:
                pass
            eng = _Engine()
            eng.model = model
            eng.processor = processor
            eng.device = next(model.parameters()).device
            eng.eos_token_id = processor.tokenizer.eos_token_id
            eng.use_gpu = use_gpu
            self._engine = eng
            logger.info("[LazyEngine] GeoChat-7B ready on %s!", eng.device)

        except Exception as exc:
            logger.exception("[LazyEngine] Failed to load model")
            self._load_error = exc
            raise

    @torch.inference_mode()
    def generate(self, image, prompt, max_new_tokens=128):
        self._ensure_loaded()
        eng = self._engine

        image = image.convert("RGB")
        inputs = eng.processor(image=image, query=prompt)

        input_ids = inputs["input_ids"].to(eng.device)
        attention_mask = inputs["attention_mask"].to(eng.device)
        dtype = torch.float16 if eng.use_gpu else torch.float32
        pixel_values = inputs["pixel_values"].to(device=eng.device, dtype=dtype).contiguous()

        # Prefill
        outputs = eng.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        # Greedy autoregressive decoding
        generated_tokens = []
        for step in range(max_new_tokens):
            if torch.isnan(next_logits).any():
                logger.error("[Generate] next_logits has NaNs at step %d! Aborting.", step)
                break

            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            token_id = next_token.item()
            if step < 5:
                top_indices = torch.topk(next_logits, min(5, next_logits.shape[-1]), dim=-1).indices[0].tolist()
                logger.info("[Generate] Step %d: token_id=%d, top_5=%s", step, token_id, top_indices)

            if eng.eos_token_id is not None and token_id == eng.eos_token_id:
                if step == 0:
                    top_indices = torch.topk(next_logits, min(5, next_logits.shape[-1]), dim=-1).indices[0].tolist()
                    logger.warning("[Generate] First token is EOS (id=%d). Top tokens: %s", token_id, top_indices)
                    # Use 2nd candidate if first is EOS to avoid empty output
                    if len(top_indices) > 1 and top_indices[0] == eng.eos_token_id:
                        next_token = torch.tensor([[top_indices[1]]], device=eng.device)
                        token_id = top_indices[1]
                    else:
                        break
                else:
                    break
            generated_tokens.append(next_token)

            if hasattr(past_key_values, "get_seq_length"):
                cache_len = past_key_values.get_seq_length()
            else:
                cache_len = past_key_values[0][0].shape[-2]

            cached_attn = torch.ones(
                (next_token.shape[0], cache_len + 1),
                dtype=attention_mask.dtype,
                device=eng.device,
            )
            outputs = eng.model(
                input_ids=next_token,
                attention_mask=cached_attn,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        if not generated_tokens:
            logger.warning("[Generate] No tokens generated.")
            return ""
        generated = torch.cat(generated_tokens, dim=1)
        raw_ids = [t.item() for t in generated_tokens]
        logger.info("[Generate] Total tokens: %d, first 15 IDs: %s", len(raw_ids), raw_ids[:15])
        decoded = eng.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if not decoded:
            raw_decoded = eng.processor.batch_decode(generated, skip_special_tokens=False)[0]
            logger.warning("[Generate] Clean decode was empty! Raw decode (with specials): %s", repr(raw_decoded[:100]))
        logger.info("[Generate] Successfully produced %d tokens. Result: %s", len(generated_tokens), repr(decoded[:120]))
        return decoded


def model_fn(model_dir, context=None):
    """Return a lazy-loading engine wrapper (defers heavy GPU work)."""
    logger.info("model_fn called with model_dir=%s", model_dir)

    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        target_model = model_dir
        logger.info("Using local model weights in: %s", model_dir)
    else:
        target_model = os.environ.get("HF_MODEL_ID", "MBZUAI/geochat-7B")
        logger.info("Using HF model ID: %s", target_model)

    engine = LazyGeoChatEngine(model_id=target_model)
    logger.info("LazyGeoChatEngine created (model will load on first request).")
    return engine


def input_fn(request_body, content_type):
    """Parse incoming JSON payload."""
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(data, engine):
    """Perform multimodal inference."""
    prompt = data.get("prompt") or data.get("inputs", "Describe this satellite image in detail.")
    image_b64 = data.get("image", "")
    sar_b64 = data.get("sar_vh") or data.get("sar", "")
    max_tokens = int(data.get("max_tokens") or data.get("parameters", {}).get("max_new_tokens", 180))

    if not image_b64:
        cuda_ok = torch.cuda.is_available()
        return {
            "generated_text": f"GeoChat-7B endpoint is active. CUDA={cuda_ok}. Prompt: {prompt}",
            "model": "MBZUAI/geochat-7B",
            "cuda_available": cuda_ok,
        }

    raw_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw_bytes))

    if sar_b64:
        sar_bytes = base64.b64decode(sar_b64)
        sar_img = Image.open(io.BytesIO(sar_bytes))
        image = sar1_to_rgb(image, sar_img)
    else:
        image = load_image_input(image)

    logger.info("[Predict] Prompt: %s | max_tokens: %d", repr(prompt[:60]), max_tokens)
    generated_text = engine.generate(
        image=image,
        prompt=prompt,
        max_new_tokens=min(max_tokens, 512),
    )
    logger.info("[Predict] Returning text length=%d: %s", len(generated_text), repr(generated_text[:100]))

    return {
        "generated_text": generated_text,
        "model": "MBZUAI/geochat-7B",
    }


def output_fn(prediction, accept_type):
    """Serialize output to JSON."""
    return json.dumps(prediction)
'''

REQUIREMENTS_TXT = ""


def get_s3_bucket(s3_client, sts_client) -> str:
    account_id = sts_client.get_caller_identity()["Account"]
    bucket_name = f"bhuvision-models-{account_id}"
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except Exception:
        s3_client.create_bucket(Bucket=bucket_name)
    return bucket_name


def create_and_upload_model_tar(s3_client, bucket_name: str) -> str:
    print("Packing model.tar.gz with custom GeoChat code...")
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        # 1. inference.py
        code_bytes = INFERENCE_PY.encode("utf-8")
        info = tarfile.TarInfo(name="code/inference.py")
        info.size = len(code_bytes)
        tar.addfile(info, io.BytesIO(code_bytes))

        # 2. requirements.txt
        req_bytes = REQUIREMENTS_TXT.encode("utf-8")
        info2 = tarfile.TarInfo(name="code/requirements.txt")
        info2.size = len(req_bytes)
        tar.addfile(info2, io.BytesIO(req_bytes))

        # 3. Add backend/geochat_src/ recursively under code/geochat/
        geochat_src_dir = os.path.join(os.path.dirname(__file__), "..", "geochat_src")
        geochat_src_dir = os.path.abspath(geochat_src_dir)

        count = 0
        for root, _, files in os.walk(geochat_src_dir):
            for f in files:
                if f.endswith((".py", ".txt", ".json")):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, geochat_src_dir)
                    arcname = f"code/geochat/{rel_path}".replace("\\", "/")
                    tar.add(full_path, arcname=arcname)
                    count += 1

        print(f"Packaged {count} GeoChat architecture files into model.tar.gz")

    tar_buffer.seek(0)
    data = tar_buffer.getvalue()
    s3_key = "geochat-7b/model.tar.gz"
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=data)
    s3_url = f"s3://{bucket_name}/{s3_key}"
    print(f"Uploaded model.tar.gz ({len(data)} bytes) to {s3_url}")
    return s3_url


def deploy(wait: bool = False):
    s3 = boto3.client("s3", region_name=REGION)
    sm = boto3.client("sagemaker", region_name=REGION)
    sts = boto3.client("sts", region_name=REGION)

    bucket_name = get_s3_bucket(s3, sts)
    model_data_url = create_and_upload_model_tar(s3, bucket_name)

    image_uri = (
        f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
        "huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04"
    )

    ts = int(time.time())
    model_name = f"geochat-custom-{ts}"
    config_name = f"geochat-config-{ts}"

    env = {
        "HF_MODEL_ID": "MBZUAI/geochat-7B",
        "HF_MODEL_TRUST_REMOTE_CODE": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
    }
    if HF_TOKEN:
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    print(f"Creating SageMaker Model: {model_name}...")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": env,
        },
        ExecutionRoleArn=ROLE_ARN,
    )
    print(f"Model created: {model_name}")

    print(f"Creating Endpoint Config: {config_name}...")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": "ml.g5.xlarge",
                "InitialInstanceCount": 1,
                "ContainerStartupHealthCheckTimeoutInSeconds": 1200,
                "ModelDataDownloadTimeoutInSeconds": 1200,
            }
        ],
    )
    print(f"Endpoint Config created: {config_name}")

    # Check existing endpoint
    endpoints = sm.list_endpoints(NameContains=ENDPOINT_NAME)["Endpoints"]
    existing = [ep for ep in endpoints if ep["EndpointName"] == ENDPOINT_NAME]

    if existing:
        ep_status = existing[0]["EndpointStatus"]
        print(f"Existing endpoint found with status: {ep_status}")
        if ep_status == "InService":
            print(f"Updating endpoint {ENDPOINT_NAME} to new config {config_name}...")
            sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
            print("Endpoint update initiated.")
        elif ep_status in ("Failed", "OutOfService"):
            print(f"Endpoint is {ep_status}, deleting and recreating...")
            sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
            time.sleep(5)
            sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
            print("Endpoint creation initiated.")
        else:
            print(f"Endpoint is in status '{ep_status}'. Waiting for it to finish before updating.")
            return
    else:
        print(f"Creating Endpoint: {ENDPOINT_NAME}...")
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        print("Endpoint creation initiated.")

    if wait:
        print(f"Waiting for endpoint {ENDPOINT_NAME} to become InService...")
        waiter = sm.get_waiter("endpoint_in_service")
        waiter.wait(EndpointName=ENDPOINT_NAME)
        print(f"Endpoint {ENDPOINT_NAME} is now InService!")


def status():
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        ep = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Endpoint Name:   {ep['EndpointName']}")
        print(f"Status:          {ep['EndpointStatus']}")
        print(f"Config:          {ep['EndpointConfigName']}")
        print(f"Created:         {ep['CreationTime']}")
        if "FailureReason" in ep:
            print(f"Failure Reason:  {ep['FailureReason']}")
    except sm.exceptions.ClientError as exc:
        print(f"Endpoint error: {exc}")


def delete():
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Deleted endpoint {ENDPOINT_NAME}")
    except Exception as exc:
        print(f"Delete failed: {exc}")


def test(image_path: str = "P2599_0003.png", prompt: str = "Describe this satellite image."):
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "prompt": prompt,
        "inputs": prompt,
        "image": img_b64,
        "max_tokens": 128,
    }
    print(f"Invoking {ENDPOINT_NAME} with prompt: '{prompt}'...")
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    print("Response:", json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SageMaker GeoChat-7B Deployment Manager")
    parser.add_argument("--deploy", action="store_true", help="Deploy or update endpoint")
    parser.add_argument("--wait", action="store_true", help="Wait for endpoint to become InService")
    parser.add_argument("--status", action="store_true", help="Check endpoint status")
    parser.add_argument("--delete", action="store_true", help="Delete endpoint")
    parser.add_argument("--test", action="store_true", help="Test endpoint with sample image")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.delete:
        delete()
    elif args.test:
        test()
    else:
        deploy(wait=args.wait)
