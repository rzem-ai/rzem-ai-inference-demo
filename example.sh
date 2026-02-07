#!/bin/bash

# CRITICAL: Set CUDA_VISIBLE_DEVICES=0 to force use of RTX 5090 (device 0)
# This must be set BEFORE running the binary, not inside Rust code
export CUDA_VISIBLE_DEVICES=0

# Set RUST_LOG=trace for detailed logging
RUST_LOG=trace cargo run --release -- compare \
    --prompt "Illustration of a cat sitting on a windowsill" \
    --lora /home/alex/Dev/Work/rzem-ai-models/rzem-ai/Retrocom1_for_Flux.safetensors \
    --seed 42