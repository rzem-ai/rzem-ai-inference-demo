//! Model loading for FLUX.1-dev pipeline components
//!
//! This module handles loading all components needed for FLUX image generation:
//! - T5-XXL text encoder (quantized GGUF)
//! - CLIP text encoder (full precision)
//! - FLUX.1-dev transformer (quantized GGUF with LoRA injection)
//! - VAE decoder (full precision)

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{clip, flux, quantized_t5};
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::{models::bpe::BPE, processors::template::TemplateProcessing, AddedToken, Tokenizer};
use tracing::{debug, info};

use crate::lora::LoraAdapter;
use candle_core::quantized::QTensor;

/// T5 text encoder for FLUX (provides main text conditioning)
pub struct T5TextEncoder {
    model: quantized_t5::T5EncoderModel,
    tokenizer: Tokenizer,
    device: Device,
    max_length: usize,
}

impl T5TextEncoder {
    /// Load quantized T5 encoder from GGUF file
    ///
    /// # Arguments
    /// * `gguf_path` - Path to t5-v1_1-xxl-encoder-Q8_0.gguf
    /// * `tokenizer_path` - Path to tokenizer directory or tokenizer.json
    /// * `device` - Device to load model on
    pub fn load<P: AsRef<Path>>(
        gguf_path: P,
        tokenizer_path: P,
        device: Device,
    ) -> Result<Self> {
        let gguf_path = gguf_path.as_ref();
        let tokenizer_path = tokenizer_path.as_ref();

        info!(path = %gguf_path.display(), "Loading T5-XXL encoder (quantized)");

        // Load tokenizer
        let tokenizer_file = if tokenizer_path.is_file() {
            tokenizer_path.to_path_buf()
        } else {
            tokenizer_path.join("tokenizer.json")
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load T5 tokenizer from {:?}: {}", tokenizer_file, e))?;

        // Load quantized model from GGUF with name mapping
        // city96's GGUF uses llama.cpp naming, we need HuggingFace naming
        let vb = MappedQVarBuilder::from_gguf(gguf_path, &device, map_llama_to_hf)?;

        // T5-XXL configuration (v1.1-xxl)
        let config_json = r#"{
            "vocab_size": 32128,
            "d_model": 4096,
            "d_kv": 64,
            "d_ff": 10240,
            "num_layers": 24,
            "num_heads": 64,
            "relative_attention_num_buckets": 32,
            "relative_attention_max_distance": 128,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-6,
            "initializer_factor": 1.0,
            "feed_forward_proj": "gated-gelu",
            "tie_word_embeddings": false,
            "is_decoder": false,
            "is_encoder_decoder": false,
            "use_cache": true,
            "pad_token_id": 0,
            "eos_token_id": 1
        }"#;
        let config: quantized_t5::Config = serde_json::from_str(config_json)?;

        let model = quantized_t5::T5EncoderModel::load(vb.into(), &config)?;

        info!("✓ T5 encoder loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
            max_length: 256, // FLUX uses 256 tokens for T5
        })
    }

    /// Encode text prompt to T5 embeddings
    ///
    /// # Arguments
    /// * `prompt` - Text prompt to encode
    ///
    /// # Returns
    /// Tensor of shape [1, seq_len, 4096] with T5 embeddings
    pub fn encode(&mut self, prompt: &str) -> Result<Tensor> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let mut tokens = encoding.get_ids().to_vec();

        // Pad or truncate to max_length
        if tokens.len() > self.max_length {
            tokens.truncate(self.max_length);
        } else {
            tokens.resize(self.max_length, 0); // Pad with 0
        }

        // Create token tensor
        let token_ids = Tensor::new(&tokens[..], &self.device)?
            .unsqueeze(0)?; // Add batch dimension

        // Encode
        let embeddings = self.model.forward(&token_ids)?;

        Ok(embeddings)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// CLIP text encoder for FLUX
pub struct ClipTextEncoder {
    model: clip::text_model::ClipTextTransformer,
    tokenizer: Tokenizer,
    device: Device,
}

impl ClipTextEncoder {
    /// Load CLIP model from safetensors file
    ///
    /// # Arguments
    /// * `model_path` - Path to model.safetensors
    /// * `tokenizer_path` - Path to tokenizer directory (with vocab.json + merges.txt)
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
        device: Device,
    ) -> Result<Self> {
        let tokenizer_path = tokenizer_path.as_ref();

        info!(path = %model_path.as_ref().display(), "Loading CLIP encoder");

        // Load tokenizer
        let tokenizer = if tokenizer_path.is_file() {
            let filename = tokenizer_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if filename == "tokenizer.json" {
                Tokenizer::from_file(tokenizer_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?
            } else if filename == "vocab.json" {
                let tokenizer_dir = tokenizer_path.parent()
                    .ok_or_else(|| anyhow::anyhow!("Could not get parent directory"))?;
                Self::load_bpe_tokenizer(tokenizer_dir)?
            } else {
                anyhow::bail!("Unknown tokenizer file: {:?}", tokenizer_path);
            }
        } else {
            // Directory - check for BPE files (vocab.json + merges.txt)
            let vocab_path = tokenizer_path.join("vocab.json");
            if vocab_path.exists() {
                Self::load_bpe_tokenizer(tokenizer_path)?
            } else {
                // Try tokenizer.json as fallback
                Tokenizer::from_file(tokenizer_path.join("tokenizer.json"))
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?
            }
        };

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_path.as_ref()],
                DType::F32,
                &device,
            )?
        };

        // FLUX CLIP model has "text_model." prefix
        let vb = vb.pp("text_model");

        // FLUX uses CLIP ViT-L configuration
        let config = clip::text_model::ClipTextConfig {
            vocab_size: 49408,
            embed_dim: 768,
            activation: clip::text_model::Activation::QuickGelu,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            pad_with: Some("<|endoftext|>".to_string()),
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 768,
        };
        let model = clip::text_model::ClipTextTransformer::new(vb, &config)?;

        info!("✓ CLIP encoder loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Load BPE tokenizer from directory with vocab.json and merges.txt
    fn load_bpe_tokenizer(tokenizer_dir: &Path) -> Result<Tokenizer> {
        let vocab_path = tokenizer_dir.join("vocab.json");
        let merges_path = tokenizer_dir.join("merges.txt");

        if !vocab_path.exists() || !merges_path.exists() {
            anyhow::bail!("BPE tokenizer requires vocab.json and merges.txt");
        }

        let _vocab = std::fs::read_to_string(&vocab_path)?;
        let _merges = std::fs::read_to_string(&merges_path)?;

        let bpe = BPE::from_file(&vocab_path.to_string_lossy(), &merges_path.to_string_lossy())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build BPE tokenizer: {}", e))?;

        let mut tokenizer = Tokenizer::new(bpe);

        // Add special tokens
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|startoftext|>", true),
            AddedToken::from("<|endoftext|>", true),
        ]);

        // Set post-processor for CLIP format
        let processor = TemplateProcessing::builder()
            .try_single("<|startoftext|> $A <|endoftext|>")
            .map_err(|e| anyhow::anyhow!("Template processing failed: {}", e))?
            .special_tokens(vec![
                ("<|startoftext|>", 49406),
                ("<|endoftext|>", 49407),
            ])
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build processor: {}", e))?;

        tokenizer.with_post_processor(Some(processor));

        // Padding and truncation to 77 tokens
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(77),
            pad_id: 49407, // <|endoftext|>
            pad_token: "<|endoftext|>".to_string(),
            ..Default::default()
        }));

        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 77,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("Failed to set truncation: {}", e))?;

        Ok(tokenizer)
    }

    /// Encode text prompt to CLIP embeddings
    ///
    /// # Arguments
    /// * `prompt` - Text prompt to encode
    ///
    /// # Returns
    /// Pooled CLIP embedding [1, 768]
    pub fn encode(&self, prompt: &str) -> Result<Tensor> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let tokens = encoding.get_ids().to_vec();

        if tokens.len() != 77 {
            anyhow::bail!("CLIP tokenization produced {} tokens, expected 77", tokens.len());
        }

        // Find EOT position (first occurrence of 49407 after the text tokens)
        let eot_position = tokens.iter()
            .position(|&t| t == 49407)
            .unwrap_or(tokens.len() - 1);

        // Create token tensor
        let token_ids = Tensor::new(&tokens[..], &self.device)?
            .unsqueeze(0)?; // Add batch dimension [1, 77]

        // Use forward_with_mask to get full hidden states
        // This bypasses the buggy argmax in the default forward() method
        let hidden_states = self.model.forward_with_mask(&token_ids, usize::MAX)?;

        // Extract hidden state at EOT position for pooled output [1, 768]
        let pooled = hidden_states.i((0, eot_position))?.unsqueeze(0)?;

        Ok(pooled)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// FLUX VAE decoder for converting latents to RGB images
pub struct VaeDecoder {
    model: flux::autoencoder::AutoEncoder,
    device: Device,
}

impl VaeDecoder {
    /// Load FLUX VAE from ae.safetensors file
    pub fn load<P: AsRef<Path>>(model_path: P, device: Device) -> Result<Self> {
        info!(path = %model_path.as_ref().display(), "Loading VAE decoder");

        // Use bf16 on CUDA for efficiency, f32 on CPU
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.as_ref()], dtype, &device)?
        };

        let config = flux::autoencoder::Config::dev();
        let model = flux::autoencoder::AutoEncoder::new(&config, vb)?;

        info!("✓ VAE decoder loaded successfully");

        Ok(Self { model, device })
    }

    /// Decode latent tensor to RGB image
    ///
    /// # Arguments
    /// * `latents` - Latent tensor [1, 16, H/8, W/8]
    ///
    /// # Returns
    /// RGB tensor [1, 3, H, W] with values in range [-1, 1]
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let image = self.model.decode(latents)?;
        Ok(image)
    }

    /// Convert decoded tensor to RGB image buffer
    ///
    /// # Arguments
    /// * `tensor` - Image tensor [1, 3, H, W] with values in [-1, 1]
    ///
    /// # Returns
    /// Vec<u8> with RGB pixel data (H * W * 3 bytes)
    pub fn tensor_to_rgb(&self, tensor: &Tensor) -> Result<Vec<u8>> {
        // Clamp to [-1, 1], scale to [0, 255]
        let image = tensor.clamp(-1f32, 1f32)?;
        let image = ((image + 1.0)? * 127.5)?;
        let image = image.to_dtype(DType::U8)?;

        // Remove batch dimension and permute to HWC
        let image = image.squeeze(0)?.permute((1, 2, 0))?;

        // Flatten to Vec<u8>
        Ok(image.flatten_all()?.to_vec1()?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// FLUX transformer model with optional LoRA injection
pub enum FluxModel {
    /// Quantized model with per-forward-pass LoRA (slow but low VRAM)
    Quantized {
        model: flux::quantized_model::Flux,
        device: Device,
    },
    /// Full precision model with pre-fused LoRA weights (fast, high VRAM)
    FullPrecision {
        model: flux::model::Flux,
        device: Device,
    },
}

impl FluxModel {
    /// Load quantized FLUX.1-dev model with per-forward-pass LoRA (slow but low VRAM)
    ///
    /// # Arguments
    /// * `gguf_path` - Path to flux1-dev-Q8_0.gguf
    /// * `loras` - List of (LoRA adapter, strength) tuples to inject
    /// * `device` - Device to load model on
    pub fn load_quantized_with_loras<P: AsRef<Path>>(
        gguf_path: P,
        loras: &[(Arc<LoraAdapter>, f32)],
        device: &Device,
    ) -> Result<Self> {
        let gguf_path = gguf_path.as_ref();

        info!(
            path = %gguf_path.display(),
            lora_count = loras.len(),
            "Loading FLUX.1-dev (quantized Q8_0)"
        );

        // Load quantized model from GGUF
        let vb = QVarBuilder::from_gguf(gguf_path, device)?;
        let cfg = flux::model::Config::dev();
        let mut model = flux::quantized_model::Flux::new(&cfg, vb)?;

        if !loras.is_empty() {
            // Prepare LoRA injection map
            let mut lora_map: HashMap<String, (Tensor, Tensor, f32)> = HashMap::new();

            info!("Preparing LoRA weights for injection:");

            for (lora, strength) in loras {
                info!(
                    name = %lora.name,
                    strength = strength,
                    weights = lora.weight_count(),
                    "  Preparing LoRA adapter"
                );

                for (layer_name, lora_weight) in &lora.weights {
                    // Map LoRA layer name to FLUX model tensor name
                    let flux_tensor_name = map_lora_to_flux_tensor(layer_name);

                    // Compute LoRA scale: (alpha / rank) * strength
                    let scale = (lora_weight.alpha / lora_weight.rank as f32) * strength;

                    // Convert LoRA tensors to f32 for computation
                    // NOTE: LoRA files store tensors as [rank, in] and [out, rank]
                    // But quantized_nn expects [in, rank] and [rank, out] for (x @ A) @ B
                    // So we transpose both tensors
                    let lora_a = lora_weight
                        .lora_down
                        .to_dtype(DType::F32)?
                        .to_device(device)?
                        .t()?;
                    let lora_b = lora_weight
                        .lora_up
                        .to_dtype(DType::F32)?
                        .to_device(device)?
                        .t()?;

                    debug!(
                        layer = %layer_name,
                        flux_name = %flux_tensor_name,
                        rank = lora_weight.rank,
                        alpha = lora_weight.alpha,
                        scale = scale,
                        "    Mapped LoRA layer"
                    );

                    lora_map.insert(flux_tensor_name, (lora_a, lora_b, scale));
                }
            }

            // Inject LoRAs into the quantized model
            info!("Injecting {} LoRA layers into FLUX model", lora_map.len());
            let injected_count = model.inject_loras(&lora_map)?;
            info!("✓ {} LoRA layers successfully injected", injected_count);
        } else {
            info!("✓ FLUX model loaded (no LoRAs)");
        }

        Ok(Self::Quantized {
            model,
            device: device.clone(),
        })
    }

    /// Load full-precision FLUX.1-dev with pre-fused LoRA weights (fast, high VRAM)
    ///
    /// This matches InvokeAI's approach: pre-compute LoRA deltas and fuse into model weights.
    ///
    /// # Arguments
    /// * `safetensors_path` - Path to flux1-dev.safetensors
    /// * `loras` - List of (LoRA adapter, strength) tuples to fuse
    /// * `device` - Device to load model on
    pub fn load_full_precision_with_fused_loras<P: AsRef<Path>>(
        safetensors_path: P,
        loras: &[(Arc<LoraAdapter>, f32)],
        device: &Device,
    ) -> Result<Self> {
        let safetensors_path = safetensors_path.as_ref();

        info!(
            path = %safetensors_path.display(),
            lora_count = loras.len(),
            "Loading FLUX.1-dev (full precision BF16)"
        );

        let dtype = DType::BF16; // Match InvokeAI's approach

        // If no LoRAs, load model directly
        if loras.is_empty() {
            info!("════════════════════════════════════════════════════════");
            info!("Loading FLUX.1-dev (full precision BF16)");
            info!("════════════════════════════════════════════════════════");
            info!("  Model: {}", safetensors_path.display());
            info!("  Device: {:?}", device);
            info!("  DType: {:?}", dtype);
            info!("  Expected VRAM: ~24GB");
            info!("");

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[safetensors_path], dtype, device)?
            };

            info!("  Building FLUX model...");
            let cfg = flux::model::Config::dev();
            let model = flux::model::Flux::new(&cfg, vb)?;

            info!("");
            info!("✓ FLUX model loaded successfully");
            info!("════════════════════════════════════════════════════════");
            info!("");

            return Ok(Self::FullPrecision {
                model,
                device: device.clone(),
            });
        }

        // With LoRAs: weight fusion not yet implemented
        // For now, load model without LoRAs to test BF16 performance
        info!("⚠ LoRA weight fusion not yet implemented");
        info!("Loading model in BF16 without LoRAs for performance testing");

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors_path], dtype, device)?
        };

        let cfg = flux::model::Config::dev();
        let model = flux::model::Flux::new(&cfg, vb)?;

        info!("✓ FLUX model loaded in BF16 (LoRA fusion pending)");
        info!("📝 TODO: Implement weight fusion to match InvokeAI's LoRA performance");

        Ok(Self::FullPrecision {
            model,
            device: device.clone(),
        })
    }

    /// Get a reference to the model for inference (works with both variants)
    pub fn model_ref(&self) -> FluxModelRef {
        match self {
            Self::Quantized { model, .. } => FluxModelRef::Quantized(model),
            Self::FullPrecision { model, .. } => FluxModelRef::FullPrecision(model),
        }
    }

    pub fn device(&self) -> &Device {
        match self {
            Self::Quantized { device, .. } => device,
            Self::FullPrecision { device, .. } => device,
        }
    }
}

/// Reference to either quantized or full-precision FLUX model
pub enum FluxModelRef<'a> {
    Quantized(&'a flux::quantized_model::Flux),
    FullPrecision(&'a flux::model::Flux),
}

/// Map LoRA layer name to FLUX model tensor name
///
/// FLUX LoRA files use naming conventions like:
/// - `lora_unet_double_blocks_0_img_attn_qkv` -> `double_blocks.0.img_attn.qkv.weight`
fn map_lora_to_flux_tensor(lora_layer: &str) -> String {
    // This implementation is simplified - it just adds .weight suffix
    // The actual tensor name mapping happens in the LoRA normalization
    // For FLUX quantized models, we use the normalized LoRA key directly
    let mut name = lora_layer.to_string();

    // Add .weight suffix if not present
    if !name.ends_with(".weight") && !name.ends_with(".bias") {
        name.push_str(".weight");
    }

    name
}

// ============================================================================
// Custom VarBuilder with name mapping support for GGUF files
// ============================================================================

/// VarBuilder for quantized tensors with name mapping support
///
/// This supports renaming tensors during load for compatibility with different
/// GGUF naming conventions (llama.cpp vs HuggingFace).
#[derive(Clone)]
struct MappedQVarBuilder {
    data: Arc<HashMap<String, Arc<QTensor>>>,
    path: Vec<String>,
    device: Device,
}

impl MappedQVarBuilder {
    /// Load GGUF file with name mapping
    ///
    /// The `rename_fn` is called for each tensor name in the GGUF file
    /// and should return the new name to use in the VarBuilder.
    fn from_gguf<P, F>(path: P, device: &Device, rename_fn: F) -> Result<Self>
    where
        P: AsRef<Path>,
        F: Fn(&str) -> String,
    {
        use candle_core::quantized::gguf_file;

        let path = path.as_ref();
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF: {}", e))?;

        let mut data = HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut file, tensor_name, device)
                .map_err(|e| anyhow::anyhow!("Failed to load tensor {}: {}", tensor_name, e))?;
            let mapped_name = rename_fn(tensor_name);
            data.insert(mapped_name, Arc::new(tensor));
        }

        debug!(tensor_count = data.len(), "Loaded tensors with name mapping");

        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    /// Create a sub-builder with a path prefix
    fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            device: self.device.clone(),
        }
    }
}

// Implement conversion to candle's VarBuilder
// Both types have identical memory layout, so we can safely transmute
impl From<MappedQVarBuilder> for QVarBuilder {
    fn from(mapped: MappedQVarBuilder) -> Self {
        // Both types have the same internal representation:
        // - data: Arc<HashMap<String, Arc<QTensor>>>
        // - path: Vec<String>
        // - device: Device
        //
        // We can safely transmute since the memory layout is identical
        unsafe { std::mem::transmute(mapped) }
    }
}

/// Map llama.cpp tensor names to HuggingFace T5 tensor names
///
/// city96's T5 GGUF files use llama.cpp naming conventions, but candle's
/// T5 implementation expects HuggingFace naming.
fn map_llama_to_hf(llama_name: &str) -> String {
    // token_embd.weight -> shared.weight
    if llama_name == "token_embd.weight" {
        return "shared.weight".to_string();
    }

    // enc.output_norm.weight -> encoder.final_layer_norm.weight
    if llama_name == "enc.output_norm.weight" {
        return "encoder.final_layer_norm.weight".to_string();
    }

    // enc.blk.{N}.* -> encoder.block.{N}.*
    if llama_name.starts_with("enc.blk.") {
        // Parse: enc.blk.{N}.{rest}
        let parts: Vec<&str> = llama_name.split('.').collect();
        if parts.len() >= 4 {
            let block_num = parts[2];
            let rest = parts[3..].join(".");

            let hf_rest = match rest.as_str() {
                // Attention weights
                "attn_k.weight" => "layer.0.SelfAttention.k.weight",
                "attn_q.weight" => "layer.0.SelfAttention.q.weight",
                "attn_v.weight" => "layer.0.SelfAttention.v.weight",
                "attn_o.weight" => "layer.0.SelfAttention.o.weight",
                "attn_rel_b.weight" => "layer.0.SelfAttention.relative_attention_bias.weight",
                "attn_norm.weight" => "layer.0.layer_norm.weight",
                // FFN weights (gated-gelu has wi_0 and wi_1)
                "ffn_gate.weight" => "layer.1.DenseReluDense.wi_0.weight",
                "ffn_up.weight" => "layer.1.DenseReluDense.wi_1.weight",
                "ffn_down.weight" => "layer.1.DenseReluDense.wo.weight",
                "ffn_norm.weight" => "layer.1.layer_norm.weight",
                _ => {
                    debug!(suffix = %rest, "Unknown tensor suffix in GGUF");
                    return llama_name.to_string();
                }
            };

            return format!("encoder.block.{}.{}", block_num, hf_rest);
        }
    }

    // Unknown pattern - return as-is
    debug!(tensor = %llama_name, "Unmapped tensor name in GGUF");
    llama_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_lora_to_flux_tensor() {
        // Test that .weight suffix is added
        assert_eq!(
            map_lora_to_flux_tensor("double_blocks.0.img_attn.qkv"),
            "double_blocks.0.img_attn.qkv.weight"
        );

        // Test single blocks pattern
        assert_eq!(
            map_lora_to_flux_tensor("single_blocks.3.linear1"),
            "single_blocks.3.linear1.weight"
        );

        // Test already has .weight suffix
        assert_eq!(
            map_lora_to_flux_tensor("double_blocks.2.qkv.weight"),
            "double_blocks.2.qkv.weight"
        );

        // Test bias suffix
        assert_eq!(
            map_lora_to_flux_tensor("some.layer.bias"),
            "some.layer.bias"
        );
    }
}
