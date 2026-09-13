# Standalone RSUniVLM VQA test

This experiment is isolated from the BHUVISION application. It does not
download checkpoints and does not modify the main backend or frontend.

## Environment

Use a Linux machine with an NVIDIA CUDA GPU. Python 3.11 is the version used
by the official RSUniVLM setup. The official implementation expects CUDA/FP16
inference; CPU and Apple M1/MPS execution are not supported by that code.

Install the official repository and its dependencies:

```bash
git clone https://github.com/xuliu-cyber/RSUniVLM.git
cd RSUniVLM
conda create -n rsunivlm python=3.11
conda activate rsunivlm
pip install --upgrade pip
pip install -e ".[train]"
cd ..
pip install -r /path/to/BHUMIVISION/experiments/rsunivlm/requirements.txt
```

Use an NVIDIA GPU with enough VRAM for the full model and runtime overhead;
the official repository does not publish a minimum VRAM value. A 24 GB GPU is
the safer starting point.

## Checkpoint and run

Download the checkpoint separately from the official RSUniVLM release and
place it on the Linux machine. This test never downloads it automatically.

```bash
export RSUNIVLM_MODEL_PATH=/models/rsunivlm/checkpoint
export RSUNIVLM_IMAGE_PATH=/data/example.png
python /path/to/BHUMIVISION/experiments/rsunivlm/test_vqa.py \
  --question "What is visible in this image?"
```

The test loads one local RGB image, runs single-image VQA, prints device/GPU
information, and prints the generated answer. It exits with an explicit error
when CUDA, the model path, the image, or the official RSUniVLM Python package
is unavailable.
