#!/bin/sh
# Fuse the chosen adapter and serve it from Ollama. Needs a llama.cpp checkout: LLAMA_CPP=/path/to/llama.cpp sh fuse.sh
# Ollama refuses Qwen3 safetensors and mlx_lm cannot export Qwen3 to GGUF, so llama.cpp's converter sits in between.
set -e
uv run python -m mlx_lm fuse --model mlx-community/Qwen3-1.7B-bf16 --adapter-path adapters/sft --save-path models/fused
uv run --no-project --python 3.12 --with transformers --with torch --with safetensors --with sentencepiece \
  --with protobuf --with numpy --with tqdm --with pyyaml --with jinja2 \
  python "$LLAMA_CPP/convert_hf_to_gguf.py" models/fused --outfile models/fused-f16.gguf --outtype f16
ollama create claims-sk -q q4_K_M -f Modelfile  # same 4-bit level as qwen3:1.7b, so the latency rows compare
rm -rf models/fused models/fused-f16.gguf       # Ollama keeps its own copy, and the disk is tight
