#!/bin/bash

# Set RUST_LOG=warn to reduce verbose LoRA logging overhead
RUST_LOG=warn cargo run --release -- compare \
    --prompt "Illustration of a cat sitting on a windowsill" \
    --lora /home/alex/Dev/Work/rzem-ai-models/rzem-ai/Retrocom1_for_Flux.safetensors \
    --seed 42