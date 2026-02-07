# FLUX.1 + LoRA Comparison Demo

Educational standalone demo showing how LoRA adapters work with quantized FLUX.1-dev models using the [Candle](https://github.com/huggingface/candle) ML framework.

## Features

- ✨ **Side-by-side comparison**: Generate with and without LoRA
- 🚀 **Quantized FLUX.1-dev**: 16GB VRAM vs 32GB full precision
- 📊 **Educational logging**: See LoRA injection process step-by-step
- 🎯 **Reproducible**: Seed control for consistent results

## What are LoRAs?

LoRA (Low-Rank Adaptation) is a technique for efficiently fine-tuning large models by training small adapter weights. Instead of modifying all model parameters, LoRAs inject low-rank updates:

```math
W' = W + (alpha/rank) * strength * (B @ A)
```

- **W**: Original FLUX transformer weight
- **A**: LoRA down-projection `[in_dim, rank]`
- **B**: LoRA up-projection `[rank, out_dim]`
- **alpha/rank**: Scaling factor (typically 1.0)
- **strength**: User multiplier (0.0 = no effect, 1.0 = full strength, 2.0 = double)

This demo uses the **quantized LoRA injection** feature from [rzem-ai/candle](https://github.com/rzem-ai/candle), which allows LoRAs to work with 8-bit quantized models for reduced VRAM usage.

## Requirements

- **GPU**: 16GB+ VRAM (CUDA or Metal)
- **Disk**: ~30GB (22GB models + cache)
- **OS**: Linux (CUDA) or macOS (Metal)
- **Rust**: 1.70+ (install from [rustup.rs](https://rustup.rs))

## Quick Start

### 1. Clone and Build

```bash
git clone https://github.com/rzem-ai/flux-lora-demo.git
cd flux-lora-demo
cargo build --release
```

### 2. Set HuggingFace Token

FLUX.1-dev is a gated model. You need to:

1. Accept the license at: [hf.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Get your token from: [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set the environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

Or add to `.env`:

```bash
cp .env.example .env
# Edit .env and add your token
```

### 3. Download Models

```bash
cargo run --release -- download
```

This downloads (~22GB, takes 10-30 minutes):

- FLUX.1-dev quantized (Q8_0, ~12GB)
- T5-XXL quantized (Q8_0, ~9GB)
- CLIP + VAE + tokenizers (~1GB)

Models are cached in `~/.cache/huggingface/hub/` and reused across runs.

### 4. Get a LoRA

**Where to find FLUX.1 LoRAs:**

#### CivitAI (largest collection)

- Browse: <https://civitai.com/models?types=LORA&baseModels=Flux.1%20D>
- Popular categories: Anime styles, photography, art styles
- Download the `.safetensors` file

#### HuggingFace

- Search: <https://huggingface.co/models?search=flux+lora>
- Example: [XLabs-AI/flux-lora-collection](https://huggingface.co/XLabs-AI/flux-lora-collection)

**Recommended LoRAs for testing:**

- **Anime style**: Search "anime" on CivitAI FLUX section
- **Realistic photography**: Search "realism" or "photography"
- **Art styles**: Search "painting" or specific artists

Download any `.safetensors` file to your local machine.

### 5. Run Comparison

```bash
cargo run --release -- compare \
  --prompt "a cat sitting on a windowsill" \
  --lora /path/to/your-lora.safetensors \
  --strength 1.0 \
  --seed 42
```

**Output**: Two images in `comparison/`:

- `baseline.png` - FLUX.1-dev without LoRA
- `with_lora.png` - FLUX.1-dev with LoRA applied

Open both side-by-side to see the LoRA effect!

## CLI Usage

### Download Models

```bash
flux-lora-demo download
```

Downloads all required models from HuggingFace Hub.

### Compare Generation

```bash
flux-lora-demo compare [OPTIONS]

Options:
  -p, --prompt <PROMPT>       Text prompt
  -l, --lora <LORA>           Path to LoRA safetensors file
  -s, --strength <STRENGTH>   LoRA strength (0.0-2.0) [default: 1.0]
      --seed <SEED>           Random seed [default: 42]
  -o, --output-dir <DIR>      Output directory [default: comparison]
```

**Examples:**

```bash
# Basic comparison
flux-lora-demo compare \
  --prompt "a cat in anime style" \
  --lora anime-lora.safetensors

# Adjust strength
flux-lora-demo compare \
  --prompt "a cat" \
  --lora anime-lora.safetensors \
  --strength 1.5

# Custom seed and output
flux-lora-demo compare \
  --prompt "a cat" \
  --lora anime-lora.safetensors \
  --seed 12345 \
  --output-dir my-comparison
```

## Example Output

```text
╔═════════════════════════════════════════════════════════╗
║     FLUX.1-dev + LoRA Side-by-Side Comparison           ║
╚═════════════════════════════════════════════════════════╝

Prompt: "a cat sitting on a windowsill"
LoRA: anime-style.safetensors
Strength: 1.0
Seed: 42

┌─ Baseline Generation (No LoRA) ─────────────────────────┐
│  [INFO] Loading quantized FLUX.1-dev from GGUF          │
│  [INFO] ✓ FLUX model loaded (no LoRAs)                  │
│  [INFO] Starting generation                             │
│  [INFO] Step 1/3: Encoding prompt with T5               │
│  [INFO] Step 2/3: Encoding prompt with CLIP             │
│  [INFO] Step 3/3: Denoising with FLUX (28 steps)        │
│  [INFO] Step 4/3: Decoding latents to RGB               │
│  [INFO] Step 5/3: Encoding PNG                          │
│  [INFO] ✓ Generation complete!                          │
│  ✓ Saved: comparison/baseline.png                       │
└─────────────────────────────────────────────────────────┘

┌─ LoRA Generation (Strength: 1.0) ───────────────────────┐
│  [INFO] Loading LoRA adapter                            │
│  [INFO]   ✓ LoRA adapter loaded successfully            │
│  [INFO]   LoRA loaded: 142 weight pairs (rank 16)       │
│  [INFO] Loading quantized FLUX.1-dev from GGUF          │
│  [INFO] Preparing LoRA weights for injection:           │
│  [INFO]   Preparing LoRA adapter                        │
│  [INFO] Injecting 142 LoRA layers into FLUX model       │
│  [INFO] ✓ 142 LoRA layers successfully injected         │
│  [INFO] ✓ Generation complete!                          │
│  ✓ Saved: comparison/with_lora.png                      │
└─────────────────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════╗
║                 Comparison Complete                     ║
╚═════════════════════════════════════════════════════════╝

📊 Results:
  • Baseline:   comparison/baseline.png
  • With LoRA:  comparison/with_lora.png

💡 Tips:
  • Open both images side-by-side to see the LoRA effect
  • Try different strengths (0.5, 1.0, 1.5) to see scaling
  • Same seed = identical baseline on repeat runs
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  FLUX.1-dev Pipeline                    │
├─────────────────────────────────────────────────────────┤
│  1. Text Encoding                                       │
│     • T5-XXL (quantized) → [1, 256, 4096] embeddings    │
│     • CLIP ViT-L → [1, 768] pooled embeddings           │
│                                                         │
│  2. Denoising (28 steps)                                │
│     • FLUX.1-dev transformer (quantized + LoRA)         │
│     • Euler sampler, normal schedule                    │
│     • Guidance scale: 3.5                               │
│                                                         │
│  3. VAE Decoding                                        │
│     • Latents [1, 16, H/8, W/8] → RGB [1, 3, H, W]      │
│                                                         │
│  4. PNG Encoding                                        │
└─────────────────────────────────────────────────────────┘
```

### Sequential Model Loading (16GB VRAM Optimization)

To fit within 16GB VRAM constraints, this demo uses **sequential loading**: models are loaded only when needed and immediately dropped after use:

```
Memory Timeline:
─────────────────────────────────────────────────────────
Time →

1. T5 Encoding
   ├─ Load T5 (9GB)
   ├─ Encode prompt → embeddings
   └─ Drop T5 ✓ (free 9GB)

2. CLIP Encoding
   ├─ Load CLIP (0.35GB)
   ├─ Encode prompt → embeddings
   └─ Drop CLIP ✓ (free 0.35GB)

3. FLUX Denoising
   ├─ Load FLUX (12GB)
   ├─ Denoise for 28 steps
   └─ Drop FLUX ✓ (free 12GB)

4. VAE Decoding
   ├─ Load VAE (0.35GB)
   ├─ Decode latents → RGB
   └─ Drop VAE ✓ (free 0.35GB)

Peak VRAM: ~16GB (vs ~18GB if all loaded simultaneously)
─────────────────────────────────────────────────────────
```

**Key benefits:**
- T5 (9GB) and FLUX (12GB) never loaded simultaneously
- Rust's RAII automatically frees memory when variables go out of scope
- Explicit `drop()` calls make memory management visible
- Same pattern used in production [rzem-ai-inference](https://github.com/rzem-ai/rzem-ai-inference)

### LoRA Injection Process

1. **Load LoRA file**: Parse safetensors to extract weight pairs (A, B matrices)
2. **Normalize layer names**: Map LoRA keys to FLUX tensor names

   ```
   lora_unet_double_blocks_0_img_attn_qkv
   → double_blocks.0.img_attn.qkv.weight
   ```

3. **Transpose tensors**: LoRA files store `[rank, in]`/`[out, rank]` but quantized_nn expects `[in, rank]`/`[rank, out]`
4. **Inject into model**: Call `model.inject_loras()` to merge LoRA weights with quantized tensors
5. **Generate**: Use modified model for denoising

### Code Structure

```
src/
├── main.rs      # CLI entry point
├── lib.rs       # Public API
├── lora.rs      # LoRA loading from safetensors
├── models.rs    # FLUX, T5, CLIP, VAE loaders
├── pipeline.rs  # Generation workflow
├── download.rs  # HuggingFace Hub downloads
└── compare.rs   # Comparison logic
```

## Troubleshooting

### "Out of memory" errors

**Option 1**: Reduce resolution (uses less VRAM):

```bash
# Edit pipeline.rs and change 1024x1024 to 512x512
```

**Option 2**: Force CPU (slow but works with any RAM):

```bash
CUDA_VISIBLE_DEVICES="" cargo run --release -- compare ...
```

### LoRA not applying / no visual difference

1. **Check LoRA compatibility** (must be FLUX.1, not SD/SDXL):

   ```bash
   # Should show FLUX layer names like "double_blocks.0.img_attn.qkv"
   strings your-lora.safetensors | grep "double_blocks"
   ```

2. **Increase strength**:

   ```bash
   cargo run --release -- compare ... --strength 1.5
   ```

3. **Check for trigger words**:
   - Many LoRAs require specific words like "anime style" or "photograph"
   - Check the model card on CivitAI/HuggingFace for recommended prompts

### "Failed to download model" errors

1. **Check HF_TOKEN**:

   ```bash
   echo $HF_TOKEN
   # Should output: hf_...
   ```

2. **Verify FLUX.1-dev access**:
   - Visit: <https://huggingface.co/black-forest-labs/FLUX.1-dev>
   - Click "Agree and access repository"

3. **Check network connection**:

   ```bash
   curl -H "Authorization: Bearer $HF_TOKEN" \
     https://huggingface.co/api/whoami
   ```

### Slow generation

**First run**: Downloads models (~22GB), can take 10-30 minutes.

**Subsequent runs**: ~30-60s per 1024x1024 image on RTX 4090 (16GB VRAM).

**With LoRA**: +5-10% overhead for injection and inference.

**CPU fallback**: 10-20x slower than GPU.

## Technical Details

### Quantization

This demo uses 8-bit quantization (Q8_0) via GGUF format:

- **FLUX.1-dev**: 32GB → 12GB (~60% reduction)
- **T5-XXL**: 20GB → 9GB (~55% reduction)
- **Total VRAM**: 16GB vs 32GB for full precision

Quality impact: Minimal (<5% difference in CLIP score).

### LoRA Support

Uses [rzem-ai/candle](https://github.com/rzem-ai/candle) fork with quantized LoRA support:

- LoRAs are kept in F32 precision (small size, no quality loss)
- Injected into quantized layers via `inject_loras()` method
- Supports multiple LoRAs with different strengths
- Clean removal via `remove_loras()` (not used in demo)

### Performance

**Hardware tested**: RTX 4090 (24GB), RTX 4080 (16GB), Apple M2 Max (64GB unified)

**Timings** (1024x1024, 28 steps):

- Model loading: 20-40s (first run only)
- T5 encoding: 0.5-1s
- CLIP encoding: 0.1-0.2s
- FLUX denoising: 25-45s
- VAE decoding: 1-2s
- **Total**: 30-50s per image

**LoRA overhead**: +1-3s for injection, +5-10% inference time.

## Credits

- **FLUX.1**: [Black Forest Labs](https://blackforestlabs.ai/)
- **Candle**: [HuggingFace](https://github.com/huggingface/candle)
- **LoRA**: [Microsoft Research](https://arxiv.org/abs/2106.09685)
- **Quantization**: [city96](https://huggingface.co/city96) (GGUF ports)
- **LoRA quantization**: [rzem-ai/candle](https://github.com/rzem-ai/candle) fork

## License

MIT License - see LICENSE file for details.

## Contributing

This is an educational demo. For production use, see the full [rzem-ai-inference](https://github.com/rzem-ai/rzem-ai-inference) application.

Issues and pull requests welcome!

## Resources

- **FLUX.1 Model**: <https://huggingface.co/black-forest-labs/FLUX.1-dev>
- **Candle Framework**: <https://github.com/huggingface/candle>
- **LoRA Paper**: <https://arxiv.org/abs/2106.09685>
- **FLUX LoRAs**: <https://civitai.com/models?types=LORA&baseModels=Flux.1%20D>
