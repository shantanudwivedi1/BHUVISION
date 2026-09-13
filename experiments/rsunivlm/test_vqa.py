"""Standalone single-image RSUniVLM VQA smoke test.

This follows the inference path in the official RSUniVLM app without
downloading checkpoints or changing the main BHUVISION application.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import NoReturn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one RSUniVLM VQA query")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("RSUNIVLM_MODEL_PATH"),
        help="Local RSUniVLM checkpoint directory (or RSUNIVLM_MODEL_PATH)",
    )
    parser.add_argument(
        "--image-path",
        default=os.environ.get("RSUNIVLM_IMAGE_PATH"),
        help="Local RGB image path (or RSUNIVLM_IMAGE_PATH)",
    )
    parser.add_argument(
        "--question",
        default=os.environ.get("RSUNIVLM_QUESTION", "What is visible in this image?"),
        help="Natural-language VQA question",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("RSUNIVLM_MODEL_NAME", "llava_qwen_gmoe"),
        help="RSUniVLM model name expected by load_pretrained_model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens for this VQA request",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    raise SystemExit(f"RSUniVLM test unavailable: {message}")


def main() -> None:
    args = parse_args()

    if not args.model_path:
        fail("set --model-path or RSUNIVLM_MODEL_PATH")
    if not args.image_path:
        fail("set --image-path or RSUNIVLM_IMAGE_PATH")

    model_path = Path(args.model_path).expanduser()
    image_path = Path(args.image_path).expanduser()
    if not model_path.exists():
        fail(f"model path does not exist: {model_path}")
    if not image_path.is_file():
        fail(f"image path does not exist or is not a file: {image_path}")

    try:
        import torch
    except ImportError as exc:
        fail(f"PyTorch is not installed: {exc}")

    if not torch.cuda.is_available():
        fail("CUDA is unavailable; run this test on a Linux NVIDIA CUDA environment")

    try:
        from PIL import Image
    except ImportError as exc:
        fail(f"Pillow is not installed: {exc}")

    try:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
    except ImportError as exc:
        fail(
            "the official RSUniVLM repository is not installed in this environment; "
            f"install it before running the test ({exc})"
        )

    device = torch.device("cuda")
    print(f"device: {device}")
    print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"cuda: {torch.version.cuda}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        fail(f"could not open image as RGB: {image_path} ({exc})")

    print(f"image: {image_path} ({image.width}x{image.height}, RGB)")
    print(f"model: {model_path}")

    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            str(model_path),
            None,
            args.model_name,
            torch_dtype="float16",
            device_map="auto",
            multimodal=True,
        )
        model.eval()

        image_tensors = process_images([image], image_processor, model.config)
        image_tensors = [tensor.to(dtype=torch.float16, device=device) for tensor in image_tensors]

        question = args.question.strip()
        if not question:
            fail("question must not be empty")
        if not question.upper().startswith("[VQA]"):
            question = f"[VQA] {question}"
        question_with_image = f"<image>\n{question}"

        conversation = copy.deepcopy(conv_templates["qwen_1_5"])
        conversation.append_message(conversation.roles[0], question_with_image)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=[image.size],
                modalities=["image"],
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new_tokens,
                granularity=0,
            )
        answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    except Exception as exc:
        fail(f"RSUniVLM inference failed: {exc}")

    print("answer:")
    print(answer)


if __name__ == "__main__":
    main()
