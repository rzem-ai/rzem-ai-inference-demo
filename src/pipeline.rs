//! Simplified FLUX.1-dev generation pipeline
//!
//! This module orchestrates the complete image generation workflow:
//! 1. Encode prompt (T5 + CLIP)
//! 2. Denoise with FLUX transformer
//! 3. VAE decode to RGB
//! 4. Save as PNG

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::flux::{self, WithForward};
use std::path::Path;
use tracing::{debug, info};

use crate::models::{ClipTextEncoder, FluxModel, T5TextEncoder, VaeDecoder};

/// Complete FLUX.1-dev generation pipeline
pub struct FluxPipeline {
    pub t5: T5TextEncoder,
    pub clip: ClipTextEncoder,
    pub vae: VaeDecoder,
    device: Device,
}

impl FluxPipeline {
    /// Create a new FLUX pipeline by loading all components
    ///
    /// # Arguments
    /// * `model_paths` - Paths to all required model files
    /// * `device` - Device to load models on
    pub fn new<P: AsRef<Path>>(
        t5_gguf: P,
        t5_tokenizer: P,
        clip_safetensors: P,
        clip_tokenizer: P,
        vae_safetensors: P,
        device: Device,
    ) -> Result<Self> {
        info!("Initializing FLUX.1-dev pipeline");

        // Load encoders
        let t5 = T5TextEncoder::load(t5_gguf, t5_tokenizer, device.clone())?;
        let clip = ClipTextEncoder::load(clip_safetensors, clip_tokenizer, device.clone())?;
        let vae = VaeDecoder::load(vae_safetensors, device.clone())?;

        info!("✓ Pipeline initialized successfully");

        Ok(Self { t5, clip, vae, device })
    }

    /// Generate an image from a text prompt using FLUX transformer
    ///
    /// # Arguments
    /// * `flux` - FLUX model (with or without LoRAs)
    /// * `prompt` - Text description of the image
    /// * `steps` - Number of denoising steps (28 for dev)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    /// PNG image data as Vec<u8>
    pub fn generate(
        &mut self,
        flux: &FluxModel,
        prompt: &str,
        steps: usize,
        width: usize,
        height: usize,
        seed: u64,
    ) -> Result<Vec<u8>> {
        info!(
            prompt_preview = %&prompt[..prompt.len().min(50)],
            steps = steps,
            size = format!("{}x{}", width, height),
            seed = seed,
            "Starting generation"
        );

        // Step 1: Encode prompt with T5
        info!("Step 1/3: Encoding prompt with T5");
        let t5_emb = self.t5.encode(prompt)?;
        info!(t5_shape = ?t5_emb.dims(), "T5 embedding shape");
        debug!(shape = ?t5_emb.dims(), "T5 embeddings");

        // Step 2: Encode prompt with CLIP
        info!("Step 2/3: Encoding prompt with CLIP");
        let clip_emb = self.clip.encode(prompt)?;
        info!(clip_shape = ?clip_emb.dims(), "CLIP embedding shape");
        debug!(shape = ?clip_emb.dims(), "CLIP embeddings");

        // Step 3: Denoise with FLUX
        info!("Step 3/3: Denoising with FLUX ({} steps)", steps);
        let latents = self.denoise(flux, &t5_emb, &clip_emb, height, width, steps, seed)?;
        debug!(shape = ?latents.dims(), "Latents generated");

        // Step 4: VAE decode
        info!("Step 4/3: Decoding latents to RGB");

        // Convert to BF16 for VAE (VAE expects BF16 on CUDA, F32 on CPU)
        let latents_for_vae = if self.device.is_cuda() {
            latents.to_dtype(DType::BF16)?
        } else {
            latents
        };

        let image = self.vae.decode(&latents_for_vae)?;
        debug!(shape = ?image.dims(), "Image decoded");

        // Step 5: Convert to PNG
        info!("Step 5/3: Encoding PNG");
        let rgb_data = self.vae.tensor_to_rgb(&image)?;
        let png_data = self.encode_png(&rgb_data, width as u32, height as u32)?;

        info!(size_kb = png_data.len() / 1024, "✓ Generation complete!");

        Ok(png_data)
    }

    /// Denoise latents using FLUX transformer with Euler sampling
    fn denoise(
        &self,
        flux: &FluxModel,
        t5_emb: &Tensor,
        clip_emb: &Tensor,
        height: usize,
        width: usize,
        steps: usize,
        seed: u64,
    ) -> Result<Tensor> {
        // Create initial noise
        let img = self.create_noise(height, width, seed)?;

        // Convert embeddings to F32 (quantized models use F32)
        let (t5_emb, clip_emb, img) = (
            t5_emb.to_dtype(DType::F32)?,
            clip_emb.to_dtype(DType::F32)?,
            img.to_dtype(DType::F32)?,
        );

        // Create sampling state
        info!(
            t5_shape = ?t5_emb.dims(),
            clip_shape = ?clip_emb.dims(),
            img_shape = ?img.dims(),
            "Creating FLUX sampling state"
        );
        let state = flux::sampling::State::new(&t5_emb, &clip_emb, &img)?;
        info!(
            txt_shape = ?state.txt.dims(),
            vec_shape = ?state.vec.dims(),
            img_shape = ?state.img.dims(),
            "FLUX state created"
        );

        // Get timestep schedule (normal/linear schedule for FLUX.1-dev)
        let timesteps = get_timesteps_normal(steps);

        // Run Euler sampling
        let guidance = 3.5; // Standard guidance for FLUX.1-dev
        let denoised = denoise_euler(flux.model(), &state, &timesteps, guidance, &self.device)?;

        // Unpack to proper shape for VAE
        let unpacked = flux::sampling::unpack(&denoised, height, width)?;

        Ok(unpacked)
    }

    /// Create initial noise tensor with given seed
    fn create_noise(&self, height: usize, width: usize, seed: u64) -> Result<Tensor> {
        // Set device seed for reproducibility
        if let Err(e) = self.device.set_seed(seed) {
            debug!(error = %e, "Could not set device seed (CPU backend)");
        }

        let noise = flux::sampling::get_noise(1, height, width, &self.device)?;
        Ok(noise.to_dtype(DType::F32)?)
    }

    /// Encode RGB data as PNG
    fn encode_png(&self, rgb_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        use image::{ImageBuffer, RgbImage};
        use std::io::Cursor;

        let img: RgbImage = ImageBuffer::from_raw(width, height, rgb_data.to_vec())
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

        let mut png_data = Cursor::new(Vec::new());
        img.write_to(&mut png_data, image::ImageFormat::Png)?;

        Ok(png_data.into_inner())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Get timestep schedule for normal/linear schedule
/// FLUX.1-dev uses a simple linear schedule from 1.0 to 0.0
fn get_timesteps_normal(steps: usize) -> Vec<f64> {
    let mut timesteps = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = 1.0 - (i as f64 / steps as f64);
        timesteps.push(t);
    }
    timesteps
}

/// Euler sampler for quantized FLUX models
/// Standard first-order ODE solver: x_{t+1} = x_t + (t_next - t_curr) * velocity
fn denoise_euler(
    model: &flux::quantized_model::Flux,
    state: &flux::sampling::State,
    timesteps: &[f64],
    guidance: f64,
    device: &Device,
) -> Result<Tensor> {
    let mut img = state.img.clone();
    let b_sz = img.dim(0)?;

    // Create guidance tensor
    let guidance_tensor = Tensor::full(guidance as f32, b_sz, device)?;

    let total_steps = timesteps.len() - 1;

    for (i, window) in timesteps.windows(2).enumerate() {
        let t_curr = window[0];
        let t_next = window[1];

        let t_vec = Tensor::full(t_curr as f32, b_sz, device)?;

        // Get velocity prediction from model
        let v = model.forward(
            &img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &t_vec,
            &state.vec,
            Some(&guidance_tensor),
        )?;

        // Euler step: x = x + dt * v
        let dt = t_next - t_curr;
        let v_scaled = (v * dt)?;
        img = (img + v_scaled)?;

        if (i + 1) % 5 == 0 || i + 1 == total_steps {
            debug!(step = i + 1, total = total_steps, "Denoising progress");
        }
    }

    Ok(img)
}
