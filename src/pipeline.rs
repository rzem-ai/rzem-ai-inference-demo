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

/// Complete FLUX.1-dev generation pipeline with sequential model loading
///
/// Loads models only when needed and drops them to minimize VRAM usage.
/// This allows running on 16GB GPUs by avoiding keeping all models loaded simultaneously.
pub struct FluxPipeline {
    // Store paths instead of loaded models
    t5_gguf_path: std::path::PathBuf,
    t5_tokenizer_path: std::path::PathBuf,
    clip_safetensors_path: std::path::PathBuf,
    clip_tokenizer_path: std::path::PathBuf,
    vae_safetensors_path: std::path::PathBuf,
    device: Device,
}

impl FluxPipeline {
    /// Create a new FLUX pipeline by storing model paths
    ///
    /// Models will be loaded sequentially during generation to minimize VRAM usage.
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
        info!("Initializing FLUX.1-dev pipeline (sequential loading)");

        // Just store paths - models will be loaded on-demand
        Ok(Self {
            t5_gguf_path: t5_gguf.as_ref().to_path_buf(),
            t5_tokenizer_path: t5_tokenizer.as_ref().to_path_buf(),
            clip_safetensors_path: clip_safetensors.as_ref().to_path_buf(),
            clip_tokenizer_path: clip_tokenizer.as_ref().to_path_buf(),
            vae_safetensors_path: vae_safetensors.as_ref().to_path_buf(),
            device,
        })
    }

    /// Encode prompt to embeddings (can be reused for multiple generations)
    ///
    /// Use this when generating multiple images with the same prompt to avoid
    /// reloading T5/CLIP each time.
    ///
    /// # Returns
    /// (T5 embeddings, CLIP embeddings)
    pub fn encode_prompt(&self, prompt: &str) -> Result<(Tensor, Tensor)> {
        // Load T5, encode, then drop
        info!("Step 1/2: Encoding prompt with T5");

        // Check memory before T5
        self.log_gpu_memory("Before T5 load");

        let t5_emb = {
            let mut t5 = T5TextEncoder::load(&self.t5_gguf_path, &self.t5_tokenizer_path, self.device.clone())?;
            self.log_gpu_memory("After T5 load");

            let emb = t5.encode(prompt)?;
            info!("  ✓ T5 encoded: shape {:?}", emb.dims());

            // Explicitly drop T5 before leaving scope
            drop(t5);
            emb
        };

        // Force CUDA to actually free T5 memory
        if self.device.is_cuda() {
            if let Err(e) = self.device.synchronize() {
                debug!("Device synchronization warning: {}", e);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // Verify T5 was actually freed
        self.log_gpu_memory("After T5 drop + sync");
        info!("");

        // Load CLIP, encode, then drop
        info!("Step 2/2: Encoding prompt with CLIP");
        let clip_emb = {
            let clip = ClipTextEncoder::load(&self.clip_safetensors_path, &self.clip_tokenizer_path, self.device.clone())?;
            self.log_gpu_memory("After CLIP load");

            let emb = clip.encode(prompt)?;
            info!("  ✓ CLIP encoded: shape {:?}", emb.dims());

            // Explicitly drop CLIP before leaving scope
            drop(clip);
            emb
        };

        // Force CUDA to actually free CLIP memory
        if self.device.is_cuda() {
            if let Err(e) = self.device.synchronize() {
                debug!("Device synchronization warning: {}", e);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // Verify CLIP was actually freed
        self.log_gpu_memory("After CLIP drop + sync");
        info!("");

        Ok((t5_emb, clip_emb))
    }

    /// Helper to log GPU memory usage
    fn log_gpu_memory(&self, label: &str) {
        if self.device.is_cuda() {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"])
                .output()
            {
                if output.status.success() {
                    let mem_info = String::from_utf8_lossy(&output.stdout);
                    if let Some(line) = mem_info.lines().next() {
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() >= 2 {
                            info!("  [{}] {} MB used, {} MB free",
                                label, parts[0].trim(), parts[1].trim());
                        }
                    }
                }
            }
        }
    }

    /// Generate image using pre-computed embeddings
    ///
    /// Use `encode_prompt()` first to get embeddings, then call this method
    /// for each generation. Useful for comparisons where the same prompt is used.
    ///
    /// # Returns
    /// PNG image data as Vec<u8>
    pub fn generate_with_embeddings(
        &self,
        flux: FluxModel,  // Take ownership to enable dropping before VAE
        t5_emb: &Tensor,
        clip_emb: &Tensor,
        steps: usize,
        width: usize,
        height: usize,
        seed: u64,
    ) -> Result<Vec<u8>> {
        info!("");
        info!("════════════════════════════════════════════════════════");
        info!("Starting generation with pre-computed embeddings");
        info!("  Size: {}x{}", width, height);
        info!("  Steps: {}", steps);
        info!("  Seed: {}", seed);
        info!("════════════════════════════════════════════════════════");
        info!("");

        // Denoise with FLUX
        info!("Step 1/3: Denoising with FLUX ({} steps)", steps);
        let latents = self.denoise(&flux, t5_emb, clip_emb, height, width, steps, seed)?;
        info!("  ✓ Latents generated: shape {:?}", latents.dims());
        info!("");

        // CRITICAL: Drop FLUX before loading VAE to free ~24GB VRAM
        drop(flux);

        // Force CUDA to synchronize and actually free the memory
        if self.device.is_cuda() {
            // Synchronize to ensure all GPU operations are complete
            if let Err(e) = self.device.synchronize() {
                debug!("Device synchronization warning: {}", e);
            }

            // Query actual GPU memory usage before VAE
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"])
                .output()
            {
                if output.status.success() {
                    let mem_info = String::from_utf8_lossy(&output.stdout);
                    if let Some(line) = mem_info.lines().next() {
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() >= 2 {
                            info!("  GPU Memory: {} MB used, {} MB free", parts[0].trim(), parts[1].trim());
                        }
                    }
                }
            }

            // Small delay to allow CUDA driver to process the deallocation
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        info!("  ✓ FLUX model unloaded");
        info!("");

        // Load VAE, decode, then drop
        info!("Step 2/3: Decoding latents to RGB");

        // Check memory before VAE load
        if self.device.is_cuda() {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"])
                .output()
            {
                if output.status.success() {
                    let mem_info = String::from_utf8_lossy(&output.stdout);
                    if let Some(line) = mem_info.lines().next() {
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() >= 2 {
                            info!("  Before VAE load: {} MB used, {} MB free", parts[0].trim(), parts[1].trim());
                        }
                    }
                }
            }
        }

        let rgb_data = {
            let vae = VaeDecoder::load(&self.vae_safetensors_path, self.device.clone())?;

            let latents_for_vae = if self.device.is_cuda() {
                latents.to_dtype(DType::BF16)?
            } else {
                latents
            };

            let image = vae.decode(&latents_for_vae)?;
            info!("  ✓ Image decoded: shape {:?}", image.dims());

            let rgb = vae.tensor_to_rgb(&image)?;
            // vae dropped here, freeing ~350MB
            rgb
        };

        // Convert to PNG
        info!("Step 3/3: Encoding PNG");
        let png_data = self.encode_png(&rgb_data, width as u32, height as u32)?;
        info!("  ✓ PNG encoded: {} KB", png_data.len() / 1024);
        info!("");

        info!("════════════════════════════════════════════════════════");
        info!("✓ Generation complete!");
        info!("════════════════════════════════════════════════════════");

        Ok(png_data)
    }

    /// Generate an image from a text prompt using FLUX transformer
    ///
    /// Uses sequential model loading to minimize VRAM usage:
    /// 1. Load T5, encode, drop T5
    /// 2. Load CLIP, encode, drop CLIP
    /// 3. Denoise with FLUX
    /// 4. Drop FLUX
    /// 5. Load VAE, decode, drop VAE
    ///
    /// This allows running on 16GB GPUs.
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
        &self,
        flux: FluxModel,  // Take ownership to enable dropping before VAE
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
            "Starting generation (sequential loading)"
        );

        // Encode prompt (loads T5, CLIP, then drops them)
        let (t5_emb, clip_emb) = self.encode_prompt(prompt)?;

        // Generate with embeddings (consumes flux)
        self.generate_with_embeddings(flux, &t5_emb, &clip_emb, steps, width, height, seed)
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

        // Convert embeddings to appropriate dtype
        let (t5_emb, clip_emb, img) = match flux {
            FluxModel::Quantized { .. } => {
                // Quantized models use F32
                (
                    t5_emb.to_dtype(DType::F32)?,
                    clip_emb.to_dtype(DType::F32)?,
                    img.to_dtype(DType::F32)?,
                )
            }
            FluxModel::FullPrecision { .. } => {
                // Full precision uses BF16
                (
                    t5_emb.to_dtype(DType::BF16)?,
                    clip_emb.to_dtype(DType::BF16)?,
                    img.to_dtype(DType::BF16)?,
                )
            }
        };

        // Create sampling state
        debug!(
            t5_shape = ?t5_emb.dims(),
            clip_shape = ?clip_emb.dims(),
            img_shape = ?img.dims(),
            "Creating FLUX sampling state"
        );
        let state = flux::sampling::State::new(&t5_emb, &clip_emb, &img)?;
        debug!(
            txt_shape = ?state.txt.dims(),
            vec_shape = ?state.vec.dims(),
            img_shape = ?state.img.dims(),
            "FLUX state created"
        );
        info!("  ✓ Sampling state created");

        // Get timestep schedule (normal/linear schedule for FLUX.1-dev)
        let timesteps = get_timesteps_normal(steps);

        // Run Euler sampling (handles both model types)
        let guidance = 3.5; // Standard guidance for FLUX.1-dev
        let denoised = match flux {
            FluxModel::Quantized { model, .. } => {
                denoise_euler_quantized(model, &state, &timesteps, guidance, &self.device)?
            }
            FluxModel::FullPrecision { model, .. } => {
                denoise_euler_full(model, &state, &timesteps, guidance, &self.device)?
            }
        };

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
fn denoise_euler_quantized(
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
            info!("  Progress: step {}/{} ({:.0}%)", i + 1, total_steps, (i + 1) as f32 / total_steps as f32 * 100.0);
        }
    }

    Ok(img)
}

/// Euler sampler for full-precision FLUX models
/// Standard first-order ODE solver: x_{t+1} = x_t + (t_next - t_curr) * velocity
fn denoise_euler_full(
    model: &flux::model::Flux,
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
            info!("  Progress: step {}/{} ({:.0}%)", i + 1, total_steps, (i + 1) as f32 / total_steps as f32 * 100.0);
        }
    }

    Ok(img)
}
