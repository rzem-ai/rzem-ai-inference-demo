//! Side-by-side LoRA comparison
//!
//! This module implements the core feature of the demo: generating two images
//! side-by-side to show the effect of LoRA adapters on FLUX.1-dev generation.

use anyhow::Result;
use candle_core::DType;
use std::path::{Path, PathBuf};
use tracing::info;

use crate::lora::LoraAdapter;
use crate::models::FluxModel;
use crate::pipeline::FluxPipeline;

/// Result of a side-by-side comparison
pub struct ComparisonResult {
    pub baseline_path: PathBuf,
    pub with_lora_path: PathBuf,
}

/// Generate two images side-by-side: baseline (no LoRA) and with LoRA
///
/// This is the main demonstration feature that shows how LoRA adapters
/// modify the model's output for the same prompt and seed.
///
/// # Arguments
/// * `pipeline` - FLUX generation pipeline (reused for both generations)
/// * `flux_gguf_path` - Path to FLUX.1-dev quantized GGUF file
/// * `prompt` - Text prompt to generate
/// * `lora_path` - Path to LoRA safetensors file
/// * `strength` - LoRA strength multiplier (0.0-2.0)
/// * `output_dir` - Directory to save comparison images
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Paths to both generated images
pub fn compare_with_without_lora<P: AsRef<Path>>(
    pipeline: &FluxPipeline,
    flux_gguf_path: P,
    prompt: &str,
    lora_path: P,
    strength: f32,
    output_dir: P,
    seed: u64,
) -> Result<ComparisonResult> {
    let flux_gguf_path = flux_gguf_path.as_ref();
    let lora_path = lora_path.as_ref();
    let output_dir = output_dir.as_ref();

    info!("╔════════════════════════════════════════════════════════╗");
    info!("║     FLUX.1-dev + LoRA Side-by-Side Comparison         ║");
    info!("╚════════════════════════════════════════════════════════╝");
    info!("");
    info!("Prompt: \"{}\"", prompt);
    info!("LoRA: {}", lora_path.file_name().unwrap().to_string_lossy());
    info!("Strength: {}", strength);
    info!("Seed: {}", seed);
    info!("");

    std::fs::create_dir_all(output_dir)?;

    let device = pipeline.device().clone();

    // ========================================================================
    // OPTIMIZE: Encode prompt once, reuse for both generations
    // ========================================================================
    info!("Encoding prompt (will be reused for both generations)");
    let (t5_emb, clip_emb) = pipeline.encode_prompt(prompt)?;
    info!("✓ Prompt encoded, embeddings ready");
    info!("");

    // ========================================================================
    // BASELINE: Generate without LoRA
    // ========================================================================
    info!("┌─ Baseline Generation (No LoRA) ──────────────────────┐");

    let baseline_png = {
        // Load FLUX, generate, then drop to free ~12GB VRAM
        let flux_baseline = FluxModel::load_with_loras(
            flux_gguf_path,
            &[], // No LoRAs
            &device,
        )?;

        let png = pipeline.generate_with_embeddings(&flux_baseline, &t5_emb, &clip_emb, 28, 1024, 1024, seed)?;

        // Explicitly drop FLUX model to free VRAM before next generation
        drop(flux_baseline);
        info!("  ✓ FLUX model unloaded (freed ~12GB VRAM)");

        png
    };

    let baseline_path = output_dir.join("baseline.png");
    std::fs::write(&baseline_path, &baseline_png)?;
    info!("└─ ✓ Saved: {}", baseline_path.display());
    info!("");

    // ========================================================================
    // WITH LORA: Generate with LoRA applied
    // ========================================================================
    info!("┌─ LoRA Generation (Strength: {}) ─────────────────┐", strength);

    let with_lora_png = {
        // Load LoRA adapter
        let lora_name = lora_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string();

        let lora = LoraAdapter::load(lora_path, lora_name, &device, DType::F32)?;

        info!("  LoRA loaded: {} weight pairs (rank {})",
            lora.weight_count(),
            lora.weights.values().next().map(|w| w.rank).unwrap_or(0)
        );

        // Load FLUX with LoRA injected
        let flux_with_lora = FluxModel::load_with_loras(
            flux_gguf_path,
            &[(lora, strength)],
            &device,
        )?;

        let png = pipeline.generate_with_embeddings(&flux_with_lora, &t5_emb, &clip_emb, 28, 1024, 1024, seed)?;

        // Explicitly drop FLUX model to free VRAM
        drop(flux_with_lora);
        info!("  ✓ FLUX model unloaded (freed ~12GB VRAM)");

        png
    };

    let with_lora_path = output_dir.join("with_lora.png");
    std::fs::write(&with_lora_path, &with_lora_png)?;
    info!("└─ ✓ Saved: {}", with_lora_path.display());
    info!("");

    // ========================================================================
    // RESULTS
    // ========================================================================
    info!("╔════════════════════════════════════════════════════════╗");
    info!("║                 Comparison Complete                    ║");
    info!("╚════════════════════════════════════════════════════════╝");
    info!("");
    info!("📊 Results:");
    info!("  • Baseline:   {}", baseline_path.display());
    info!("  • With LoRA:  {}", with_lora_path.display());
    info!("");
    info!("💡 Tips:");
    info!("  • Open both images side-by-side to see the LoRA effect");
    info!("  • Try different strengths (0.5, 1.0, 1.5) to see scaling");
    info!("  • Same seed = identical baseline on repeat runs");

    Ok(ComparisonResult {
        baseline_path,
        with_lora_path,
    })
}
