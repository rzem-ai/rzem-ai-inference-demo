# FLUX.1 + LoRA Demo - Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- **GPU**: 16GB+ VRAM (NVIDIA with CUDA, or Apple Silicon with Metal)
- **Rust**: Install from [rustup.rs](https://rustup.rs)
- **HuggingFace Account**: For gated model access

## Step 1: Clone and Build (2 min)

```bash
git clone https://github.com/rzem-ai/flux-lora-demo.git
cd flux-lora-demo
cargo build --release
```

## Step 2: Setup HuggingFace Token (1 min)

1. Go to <https://huggingface.co/black-forest-labs/FLUX.1-dev>
2. Click "Agree and access repository"
3. Get your token from <https://huggingface.co/settings/tokens>
4. Set environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

## Step 3: Download Models (10-30 min, ~22GB)

```bash
cargo run --release -- download
```

**Note**: This only needs to be done once. Models are cached for future runs.

## Step 4: Get a LoRA

Visit [CivitAI FLUX LoRAs](https://civitai.com/models?types=LORA&baseModels=Flux.1%20D) and download any `.safetensors` file.

**Quick test LoRA**: Search for "anime" and download any popular anime-style LoRA.

## Step 5: Run Comparison (1-2 min per generation)

```bash
cargo run --release -- compare \
  --prompt "a cat sitting on a windowsill" \
  --lora /path/to/your-lora.safetensors \
  --strength 1.0
```

**Output**: Two images in `comparison/`:

- `baseline.png` - Without LoRA
- `with_lora.png` - With LoRA

Open both side-by-side to see the effect!

## Troubleshooting

### Out of Memory

Reduce resolution by editing `src/pipeline.rs` line 78:

```rust
// Change from 1024x1024 to 512x512
let baseline_png = pipeline.generate(&flux_baseline, prompt, 28, 512, 512, seed)?;
```

### No Visual Difference

1. Check LoRA is for FLUX.1 (not SD/SDXL)
2. Increase strength: `--strength 1.5`
3. Check if LoRA requires trigger words in prompt

### Slow Generation

First run: Downloads models (~22GB)
Subsequent runs: ~30-60s per image on RTX 4090

## Next Steps

- Try different LoRAs from [CivitAI](https://civitai.com/models?types=LORA&baseModels=Flux.1%20D)
- Experiment with strength values (0.5, 1.0, 1.5, 2.0)
- Read the full [README.md](README.md) for technical details
- Explore the code to learn about LoRA injection

## Support

- Issues: <https://github.com/rzem-ai/flux-lora-demo/issues>
- Documentation: [README.md](README.md)
