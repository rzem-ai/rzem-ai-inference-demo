//! LoRA (Low-Rank Adaptation) weight loading
//!
//! LoRA allows fine-tuning large models by training small adapter weights
//! that are merged with the base model: W' = W + (alpha/rank) * strength * (B @ A)
//!
//! This module handles loading LoRA weights from safetensors files and preparing
//! them for injection into quantized FLUX models.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Individual LoRA weight pair for a single layer
#[derive(Debug)]
pub struct LoraWeight {
    /// Layer name this weight applies to (e.g., "double_blocks.0.img_attn.qkv")
    pub layer_name: String,
    /// Down projection matrix (A) - reduces dimensionality [rank, in_features]
    pub lora_down: Tensor,
    /// Up projection matrix (B) - restores dimensionality [out_features, rank]
    pub lora_up: Tensor,
    /// Alpha scaling factor (defaults to rank if not specified)
    pub alpha: f32,
    /// Rank (dimension of the low-rank decomposition)
    pub rank: usize,
}

/// LoRA adapter containing all weights for a fine-tuned model
#[derive(Debug)]
pub struct LoraAdapter {
    /// Display name
    pub name: String,
    /// All weight pairs keyed by normalized layer name
    pub weights: HashMap<String, LoraWeight>,
}

impl LoraAdapter {
    /// Load a LoRA adapter from a safetensors file
    ///
    /// # Arguments
    /// * `path` - Path to the .safetensors file
    /// * `name` - Display name for this LoRA
    /// * `device` - Device to load tensors onto
    /// * `dtype` - Data type for tensors (usually F32 or BF16)
    pub fn load<P: AsRef<Path>>(
        path: P,
        name: String,
        device: &Device,
        dtype: DType,
    ) -> Result<Arc<Self>> {
        let path = path.as_ref();
        info!(
            path = %path.display(),
            "Loading LoRA adapter"
        );

        let file_data = std::fs::read(path)
            .with_context(|| format!("Failed to read LoRA file: {}", path.display()))?;

        let tensors = SafeTensors::deserialize(&file_data)
            .with_context(|| format!("Failed to parse safetensors: {}", path.display()))?;

        let mut weights = HashMap::new();
        let mut alpha_values: HashMap<String, f32> = HashMap::new();
        let mut down_tensors: HashMap<String, Tensor> = HashMap::new();
        let mut up_tensors: HashMap<String, Tensor> = HashMap::new();

        // First pass: collect all tensors and alphas
        for (key, _) in tensors.tensors() {
            if key.ends_with(".alpha") {
                // Extract alpha value
                let tensor = load_tensor_from_safetensors(&tensors, &key, device, dtype)?;
                // Convert to F32 and move to CPU to safely extract scalar value
                let alpha = tensor
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_scalar::<f32>()?;
                let base_name = key.strip_suffix(".alpha").unwrap();
                let normalized = normalize_lora_key(base_name);
                debug!(
                    key = %key,
                    alpha = alpha,
                    normalized = %normalized,
                    "Extracted alpha value"
                );
                alpha_values.insert(normalized, alpha);
            } else if key.contains(".lora_down.") || key.contains(".lora_A.") {
                let tensor = load_tensor_from_safetensors(&tensors, &key, device, dtype)?;
                let base_name = extract_lora_base_name(&key);
                let normalized = normalize_lora_key(&base_name);
                down_tensors.insert(normalized, tensor);
            } else if key.contains(".lora_up.") || key.contains(".lora_B.") {
                let tensor = load_tensor_from_safetensors(&tensors, &key, device, dtype)?;
                let base_name = extract_lora_base_name(&key);
                let normalized = normalize_lora_key(&base_name);
                up_tensors.insert(normalized, tensor);
            }
        }

        // Second pass: pair up the weights
        for (layer_name, lora_down) in down_tensors {
            if let Some(lora_up) = up_tensors.remove(&layer_name) {
                // Determine rank from the down tensor shape
                let rank = lora_down.dims()[0];

                // Get alpha, defaulting to rank if not specified
                let alpha = alpha_values.get(&layer_name).copied().unwrap_or(rank as f32);

                debug!(
                    layer = %layer_name,
                    rank = rank,
                    alpha = alpha,
                    "Loaded LoRA weight pair"
                );

                weights.insert(
                    layer_name.clone(),
                    LoraWeight {
                        layer_name: layer_name.clone(),
                        lora_down,
                        lora_up,
                        alpha,
                        rank,
                    },
                );
            } else {
                warn!(layer = %layer_name, "LoRA down tensor without matching up tensor");
            }
        }

        // Warn about any orphaned up tensors
        for layer_name in up_tensors.keys() {
            warn!(layer = %layer_name, "LoRA up tensor without matching down tensor");
        }

        info!(
            path = %path.display(),
            weight_pairs = weights.len(),
            "✓ LoRA adapter loaded successfully"
        );

        Ok(Arc::new(Self { name, weights }))
    }

    /// Get the number of weight pairs in this adapter
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }
}

/// Load a tensor from safetensors and convert to target dtype
fn load_tensor_from_safetensors(
    tensors: &SafeTensors,
    key: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let view = tensors
        .tensor(key)
        .with_context(|| format!("Tensor not found: {}", key))?;

    // Convert safetensors dtype to candle dtype
    let st_dtype = view.dtype();
    let candle_dtype = match st_dtype {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        _ => anyhow::bail!("Unsupported tensor dtype: {:?}", st_dtype),
    };

    // Create tensor from raw data
    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data();

    // Load as original dtype first
    let tensor = match candle_dtype {
        DType::F32 => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Tensor::from_vec(floats, shape.as_slice(), device)?
        }
        DType::F16 => {
            let halfs: Vec<half::f16> = data
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                .collect();
            Tensor::from_vec(halfs, shape.as_slice(), device)?
        }
        DType::BF16 => {
            let bhalfs: Vec<half::bf16> = data
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]))
                .collect();
            Tensor::from_vec(bhalfs, shape.as_slice(), device)?
        }
        _ => unreachable!(),
    };

    // Convert to target dtype if different
    if candle_dtype != dtype {
        Ok(tensor.to_dtype(dtype)?)
    } else {
        Ok(tensor)
    }
}

/// Extract the base layer name from a LoRA key
/// e.g., "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight" -> "lora_unet_double_blocks_0_img_attn_qkv"
fn extract_lora_base_name(key: &str) -> String {
    // Remove the .lora_down.weight, .lora_up.weight, .lora_A.weight, .lora_B.weight suffix
    let key = key.strip_suffix(".weight").unwrap_or(key);

    if let Some(pos) = key.rfind(".lora_down") {
        key[..pos].to_string()
    } else if let Some(pos) = key.rfind(".lora_up") {
        key[..pos].to_string()
    } else if let Some(pos) = key.rfind(".lora_A") {
        key[..pos].to_string()
    } else if let Some(pos) = key.rfind(".lora_B") {
        key[..pos].to_string()
    } else {
        key.to_string()
    }
}

/// Normalize LoRA key to FLUX transformer layer name
/// Converts various LoRA naming conventions to the internal FLUX layer names
pub fn normalize_lora_key(key: &str) -> String {
    // Common FLUX LoRA patterns:
    // "lora_unet_double_blocks_0_img_attn_qkv" -> "double_blocks.0.img_attn.qkv"
    // "transformer.transformer_blocks.0.attn.to_q" -> "transformer_blocks.0.attn.to_q"

    let mut normalized = key.to_string();

    // Remove common prefixes
    for prefix in ["lora_unet_", "lora_te_", "lora_te1_", "lora_te2_", "transformer."] {
        if normalized.starts_with(prefix) {
            normalized = normalized[prefix.len()..].to_string();
        }
    }

    // Convert underscores to dots for block indices
    // double_blocks_0_img_attn -> double_blocks.0.img_attn
    let parts: Vec<&str> = normalized.split('_').collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < parts.len() {
        let part = parts[i];

        // Check if this is a block name followed by a number
        if i + 1 < parts.len() && parts[i + 1].parse::<u32>().is_ok() {
            // Combine: "double_blocks" + "0" -> "double_blocks.0"
            result.push(format!("{}.{}", part, parts[i + 1]));
            i += 2;
        } else {
            result.push(part.to_string());
            i += 1;
        }
    }

    result.join(".")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_lora_key() {
        // Test FLUX double blocks pattern
        assert_eq!(
            normalize_lora_key("lora_unet_double_blocks_0_img_attn_qkv"),
            "double.blocks.0.img.attn.qkv"
        );

        // Test single blocks pattern
        assert_eq!(
            normalize_lora_key("lora_unet_single_blocks_5_txt_mlp_fc1"),
            "single.blocks.5.txt.mlp.fc1"
        );

        // Test with transformer prefix
        assert_eq!(
            normalize_lora_key("transformer.transformer_blocks.0.attn.to_q"),
            "transformer.blocks.0.attn.to.q"
        );

        // Test already normalized (with dots)
        assert_eq!(
            normalize_lora_key("double.blocks.3.img_attn.proj"),
            "double.blocks.3.img.attn.proj"
        );
    }

    #[test]
    fn test_extract_lora_base_name() {
        assert_eq!(
            extract_lora_base_name("lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight"),
            "lora_unet_double_blocks_0_img_attn_qkv"
        );

        assert_eq!(
            extract_lora_base_name("some_layer.lora_up.weight"),
            "some_layer"
        );

        assert_eq!(
            extract_lora_base_name("layer.lora_A.weight"),
            "layer"
        );

        assert_eq!(
            extract_lora_base_name("layer.lora_B.weight"),
            "layer"
        );
    }
}
