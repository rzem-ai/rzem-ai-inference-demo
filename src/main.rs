//! CLI entry point for FLUX.1-dev + LoRA comparison demo

use anyhow::Result;
use clap::{Parser, Subcommand};
use flux_lora_demo::compare::compare_with_without_lora;
use flux_lora_demo::download::ModelDownloader;
use flux_lora_demo::pipeline::FluxPipeline;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "flux-lora-demo")]
#[command(author = "rzem-ai")]
#[command(version = "0.1.0")]
#[command(about = "FLUX.1-dev + LoRA comparison demo", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download required models (~22GB)
    ///
    /// Downloads from HuggingFace Hub:
    /// - FLUX.1-dev quantized (Q8_0, ~12GB)
    /// - T5-XXL quantized (Q8_0, ~9GB)
    /// - CLIP encoder + VAE + tokenizers (~1GB)
    ///
    /// Requires HF_TOKEN environment variable for gated models.
    /// Get token from: https://huggingface.co/settings/tokens
    Download,

    /// Compare generation with/without LoRA (side-by-side)
    ///
    /// Generates two images:
    /// 1. Baseline: FLUX.1-dev without LoRA
    /// 2. With LoRA: FLUX.1-dev with LoRA adapter applied
    ///
    /// Both use the same prompt and seed for direct comparison.
    Compare {
        /// Text prompt
        #[arg(short, long)]
        prompt: String,

        /// Path to LoRA safetensors file
        #[arg(short, long)]
        lora: PathBuf,

        /// LoRA strength (0.0-2.0)
        ///
        /// - 0.0 = no effect
        /// - 1.0 = full strength (recommended)
        /// - 1.5-2.0 = amplified effect
        #[arg(short, long, default_value = "1.0")]
        strength: f32,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Output directory
        #[arg(short, long, default_value = "comparison")]
        output_dir: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Download => {
            println!();
            println!("╔════════════════════════════════════════════════════════╗");
            println!("║     FLUX.1-dev Model Downloader                        ║");
            println!("╚════════════════════════════════════════════════════════╝");
            println!();
            println!("This will download ~22GB of models from HuggingFace Hub:");
            println!("  • FLUX.1-dev quantized (Q8_0) - ~12GB");
            println!("  • T5-XXL quantized (Q8_0) - ~9GB");
            println!("  • CLIP encoder + VAE + tokenizers - ~1GB");
            println!();
            println!("⚠️  FLUX.1-dev is a gated model. You need to:");
            println!("  1. Accept the license at: https://huggingface.co/black-forest-labs/FLUX.1-dev");
            println!("  2. Set HF_TOKEN environment variable with your token");
            println!("  3. Get token from: https://huggingface.co/settings/tokens");
            println!();

            // Check for HF_TOKEN
            if std::env::var("HF_TOKEN").is_err() {
                eprintln!("❌ Error: HF_TOKEN environment variable not set");
                eprintln!();
                eprintln!("Please set your HuggingFace token:");
                eprintln!("  export HF_TOKEN=hf_your_token_here");
                eprintln!();
                std::process::exit(1);
            }

            let downloader = ModelDownloader::new()?;
            let paths = downloader.download_all().await?;

            println!();
            println!("✓ All models downloaded successfully!");
            println!();
            println!("Model locations:");
            println!("  FLUX:  {}", paths.flux_gguf.display());
            println!("  T5:    {}", paths.t5_gguf.display());
            println!("  CLIP:  {}", paths.clip_safetensors.display());
            println!("  VAE:   {}", paths.vae_safetensors.display());
            println!();
            println!("Next steps:");
            println!("  1. Download a FLUX LoRA from CivitAI or HuggingFace");
            println!("  2. Run: flux-lora-demo compare --prompt \"...\" --lora path/to/lora.safetensors");
            println!();
        }

        Commands::Compare {
            prompt,
            lora,
            strength,
            seed,
            output_dir,
        } => {
            // Validate LoRA file exists
            if !lora.exists() {
                eprintln!("❌ Error: LoRA file not found: {}", lora.display());
                eprintln!();
                eprintln!("Where to find FLUX LoRAs:");
                eprintln!("  • CivitAI: https://civitai.com/models?types=LORA&baseModels=Flux.1%20D");
                eprintln!("  • HuggingFace: https://huggingface.co/models?search=flux+lora");
                eprintln!();
                std::process::exit(1);
            }

            // Validate strength range
            if strength < 0.0 || strength > 2.0 {
                eprintln!("⚠️  Warning: Strength {} is outside recommended range [0.0, 2.0]", strength);
            }

            println!();
            println!("════════════════════════════════════════════════════════");
            println!("🚀 Initializing FLUX.1-dev Pipeline (Full Precision BF16)");
            println!("════════════════════════════════════════════════════════");
            println!();

            // Detect and display all available GPUs
            println!("🔍 Detecting GPU devices...");
            println!();

            #[cfg(feature = "cuda")]
            {
                // Try to get GPU info using nvidia-smi
                let gpu_info = std::process::Command::new("nvidia-smi")
                    .args(&["--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"])
                    .output();

                if let Ok(output) = gpu_info {
                    if output.status.success() {
                        let gpu_list = String::from_utf8_lossy(&output.stdout);
                        println!("Available GPUs:");
                        for line in gpu_list.lines() {
                            let parts: Vec<&str> = line.split(',').collect();
                            if parts.len() >= 4 {
                                println!("  GPU {}: {} - {}MB total, {}MB free",
                                    parts[0].trim(),
                                    parts[1].trim(),
                                    parts[2].trim(),
                                    parts[3].trim()
                                );
                            }
                        }
                        println!();
                    }
                }
            }

            // Initialize device - explicitly use GPU 0
            println!("📌 Selecting GPU device 0...");
            let device = candle_core::Device::new_cuda(0)
                .or_else(|_| candle_core::Device::cuda_if_available(0))?;

            println!("✓ Using device: {:?}", device);

            // Verify we got the right device
            match &device {
                candle_core::Device::Cuda(cuda_device) => {
                    println!("  CUDA Device ID: {:?}", cuda_device);
                    println!("  Expected: RTX 5090 (32GB VRAM)");
                }
                candle_core::Device::Metal(metal_device) => {
                    println!("  Metal Device ID: {:?}", metal_device);
                }
                candle_core::Device::Cpu => {
                    println!("  ⚠️  WARNING: Using CPU (will be very slow!)");
                }
            }
            println!();

            // Download models if needed
            let downloader = ModelDownloader::new()?;
            let paths = downloader.download_all().await?;

            // Create pipeline
            let mut pipeline = FluxPipeline::new(
                &paths.t5_gguf,
                &paths.t5_tokenizer(),
                &paths.clip_safetensors,
                &paths.clip_tokenizer(),
                &paths.vae_safetensors,
                device,
            )?;

            println!();

            // Run comparison
            let result = compare_with_without_lora(
                &pipeline,
                &paths.flux_full,  // Use full-precision model
                &prompt,
                &lora,
                strength,
                &output_dir,
                seed,
            )?;

            println!();
            println!("════════════════════════════════════════════════════════");
            println!();
            println!("✨ Success! Open these images side-by-side:");
            println!("  Baseline:  {}", result.baseline_path.display());
            println!("  With LoRA: {}", result.with_lora_path.display());
            println!();
        }
    }

    Ok(())
}
