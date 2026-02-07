//! Model downloader for HuggingFace Hub
//!
//! This module handles downloading the required models for FLUX.1-dev generation:
//! - FLUX.1-dev quantized (~12GB GGUF)
//! - T5-XXL quantized (~9GB GGUF)
//! - CLIP encoder + VAE + tokenizers (~1GB)

use anyhow::{Context, Result};
use hf_hub::api::tokio::Api;
use std::path::PathBuf;
use tracing::info;

/// Model downloader that caches models using HuggingFace Hub
pub struct ModelDownloader {
    api: Api,
}

impl ModelDownloader {
    /// Create a new model downloader
    ///
    /// Uses HF_TOKEN environment variable if set for gated models
    pub fn new() -> Result<Self> {
        let api = Api::new().context("Failed to create HuggingFace API client")?;
        Ok(Self { api })
    }

    /// Download all required models for FLUX.1-dev generation
    ///
    /// Total download size: ~46GB
    /// - FLUX.1-dev full precision: ~24GB
    /// - FLUX.1-dev Q8_0 (optional): ~12GB
    /// - T5-XXL Q8_0: ~9GB
    /// - CLIP + VAE + tokenizers: ~1GB
    ///
    /// Returns paths to downloaded models
    pub async fn download_all(&self) -> Result<ModelPaths> {
        info!("Downloading FLUX.1-dev pipeline models (full precision mode)");
        info!("This may take 20-40 minutes depending on your connection");

        // Download in parallel
        let (flux_full, flux_gguf, t5_gguf, clip_safetensors, vae_safetensors, tokenizer_dir) = tokio::try_join!(
            self.download_flux_dev_full(),
            self.download_flux_dev_gguf(),
            self.download_t5_gguf(),
            self.download_clip(),
            self.download_vae(),
            self.download_tokenizers()
        )?;

        info!("✓ All models downloaded successfully!");

        Ok(ModelPaths {
            flux_full,
            flux_gguf,
            t5_gguf,
            clip_safetensors,
            vae_safetensors,
            tokenizer_dir,
        })
    }

    /// Download FLUX.1-dev quantized GGUF (~12GB)
    pub async fn download_flux_dev_gguf(&self) -> Result<PathBuf> {
        info!("Downloading FLUX.1-dev quantized (Q8_0, ~12GB)");

        let repo = self.api.repo(hf_hub::Repo::model(
            "city96/FLUX.1-dev-gguf".to_string(),
        ));
        let path = repo
            .get("flux1-dev-Q8_0.gguf")
            .await
            .context("Failed to download FLUX.1-dev GGUF")?;

        info!("  ✓ FLUX.1-dev downloaded: {}", path.display());
        Ok(path)
    }

    /// Download FLUX.1-dev full precision safetensors (~24GB)
    ///
    /// This is the full BF16 model for high-performance inference.
    /// Use this instead of GGUF for 10x faster LoRA generation.
    pub async fn download_flux_dev_full(&self) -> Result<PathBuf> {
        info!("Downloading FLUX.1-dev full precision (BF16, ~24GB)");

        let repo = self.api.repo(hf_hub::Repo::model(
            "black-forest-labs/FLUX.1-dev".to_string(),
        ));
        let path = repo
            .get("flux1-dev.safetensors")
            .await
            .context("Failed to download FLUX.1-dev safetensors")?;

        info!("  ✓ FLUX.1-dev full precision downloaded: {}", path.display());
        Ok(path)
    }

    /// Download T5-XXL quantized GGUF (~9GB)
    pub async fn download_t5_gguf(&self) -> Result<PathBuf> {
        info!("Downloading T5-XXL quantized (Q8_0, ~9GB)");

        let repo = self.api.repo(hf_hub::Repo::model(
            "city96/t5-v1_1-xxl-encoder-gguf".to_string(),
        ));
        let path = repo
            .get("t5-v1_1-xxl-encoder-Q8_0.gguf")
            .await
            .context("Failed to download T5 GGUF")?;

        info!("  ✓ T5-XXL downloaded: {}", path.display());
        Ok(path)
    }

    /// Download CLIP encoder safetensors (~350MB)
    pub async fn download_clip(&self) -> Result<PathBuf> {
        info!("Downloading CLIP encoder (~350MB)");

        let repo = self.api.repo(hf_hub::Repo::model(
            "black-forest-labs/FLUX.1-dev".to_string(),
        ));
        let path = repo
            .get("text_encoder/model.safetensors")
            .await
            .context("Failed to download CLIP model")?;

        info!("  ✓ CLIP encoder downloaded: {}", path.display());
        Ok(path)
    }

    /// Download VAE decoder safetensors (~350MB)
    pub async fn download_vae(&self) -> Result<PathBuf> {
        info!("Downloading VAE decoder (~350MB)");

        let repo = self.api.repo(hf_hub::Repo::model(
            "black-forest-labs/FLUX.1-dev".to_string(),
        ));
        let path = repo
            .get("ae.safetensors")
            .await
            .context("Failed to download VAE model")?;

        info!("  ✓ VAE decoder downloaded: {}", path.display());
        Ok(path)
    }

    /// Download tokenizers (~2MB total)
    pub async fn download_tokenizers(&self) -> Result<PathBuf> {
        info!("Downloading tokenizers (~2MB)");

        let repo = self.api.repo(hf_hub::Repo::model(
            "black-forest-labs/FLUX.1-dev".to_string(),
        ));

        // Download CLIP tokenizer files (BPE format)
        let vocab_json = repo
            .get("tokenizer/vocab.json")
            .await
            .context("Failed to download CLIP vocab.json")?;

        let _merges_txt = repo
            .get("tokenizer/merges.txt")
            .await
            .context("Failed to download CLIP merges.txt")?;

        // Download T5 tokenizer (unified format)
        let _t5_tokenizer = repo
            .get("tokenizer_2/tokenizer.json")
            .await
            .context("Failed to download T5 tokenizer")?;

        // Return the parent directory that contains both tokenizer/ and tokenizer_2/
        let tokenizer_dir = vocab_json
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| anyhow::anyhow!("Could not determine tokenizer directory"))?
            .to_path_buf();

        info!("  ✓ Tokenizers downloaded: {}", tokenizer_dir.display());
        Ok(tokenizer_dir)
    }
}

/// Paths to all downloaded models
pub struct ModelPaths {
    pub flux_full: PathBuf,      // Full precision BF16 (fast, 24GB)
    pub flux_gguf: PathBuf,       // Quantized Q8_0 (slow, 12GB)
    pub t5_gguf: PathBuf,
    pub clip_safetensors: PathBuf,
    pub vae_safetensors: PathBuf,
    pub tokenizer_dir: PathBuf,
}

impl ModelPaths {
    /// Get path to T5 tokenizer (tokenizer_2/tokenizer.json)
    pub fn t5_tokenizer(&self) -> PathBuf {
        self.tokenizer_dir.join("tokenizer_2/tokenizer.json")
    }

    /// Get path to CLIP tokenizer directory (contains vocab.json + merges.txt)
    pub fn clip_tokenizer(&self) -> PathBuf {
        self.tokenizer_dir.join("tokenizer")
    }

    /// Get path to CLIP vocab.json
    pub fn clip_vocab(&self) -> PathBuf {
        self.tokenizer_dir.join("tokenizer/vocab.json")
    }

    /// Get path to CLIP merges.txt
    pub fn clip_merges(&self) -> PathBuf {
        self.tokenizer_dir.join("tokenizer/merges.txt")
    }
}
