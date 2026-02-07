//! FLUX.1-dev + LoRA Comparison Demo
//!
//! Educational standalone demo showing how LoRA adapters work with
//! quantized FLUX.1-dev models using the Candle ML framework.
//!
//! ## Features
//!
//! - **Side-by-side comparison**: Generate with and without LoRA
//! - **Quantized models**: 16GB VRAM vs 32GB full precision
//! - **Educational logging**: See LoRA injection process step-by-step
//! - **Reproducible**: Seed control for consistent results
//!
//! ## Usage
//!
//! ```rust,ignore
//! use flux_lora_demo::download::ModelDownloader;
//! use flux_lora_demo::pipeline::FluxPipeline;
//! use flux_lora_demo::compare::compare_with_without_lora;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Download models
//!     let downloader = ModelDownloader::new()?;
//!     let paths = downloader.download_all().await?;
//!
//!     // Create pipeline
//!     let device = candle_core::Device::cuda_if_available(0)?;
//!     let mut pipeline = FluxPipeline::new(
//!         &paths.t5_gguf,
//!         &paths.t5_tokenizer(),
//!         &paths.clip_safetensors,
//!         &paths.clip_tokenizer(),
//!         &paths.vae_safetensors,
//!         device,
//!     )?;
//!
//!     // Compare with/without LoRA
//!     compare_with_without_lora(
//!         &mut pipeline,
//!         &paths.flux_gguf,
//!         "a cat sitting on a windowsill",
//!         std::path::Path::new("path/to/lora.safetensors"),
//!         1.0,
//!         std::path::Path::new("comparison/"),
//!         42,
//!     )?;
//!
//!     Ok(())
//! }
//! ```

pub mod compare;
pub mod download;
pub mod lora;
pub mod models;
pub mod pipeline;
