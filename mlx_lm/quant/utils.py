# Copyright © 2025 Apple Inc.

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from ..models.bitlinear_layers import BitLinear

QUANT_LINEAR_MAPPING = {
    'bitnet': BitLinear,
}

def load_data(tokenizer, num_samples: int, sequence_length: int) -> mx.array:
    save_dir = Path.home() / ".cache/mlx-lm/calibration_v5.txt"
    if not save_dir.exists():
        from urllib import request

        save_dir.parent.mkdir(parents=True, exist_ok=True)
        url = "https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw/571fda718462de863e5a0171078c175420c7649a/calibration_data_v5_rc.txt"
        request.urlretrieve(url, save_dir)
    with open(save_dir) as fid:
        texts = fid.read()
    tokens = tokenizer.encode(texts, return_tensors="mlx")[0]

    # select random non-overlapping chunks
    tokens = tokens[: (tokens.size // sequence_length) * sequence_length]
    tokens = tokens.reshape(-1, sequence_length)
    segments = mx.random.permutation(tokens.shape[0])
    if num_samples > 0:
        segments = segments[:num_samples]
    return tokens[segments]

def replace_linear_with_quant_linear(model, quant_method = "bitnet", modules_to_not_convert=None, fuse_qkv=False):
    quantize_layers = []
    for name, module in model.named_modules():     
        if modules_to_not_convert is None:
            modules_to_not_convert = []
        # Replace nn.Linear layers, but skip 'lm_head'
        if name not in modules_to_not_convert and isinstance(module, nn.Linear):
            old_weight = module.weight
            out_features, in_features = old_weight.shape
            bias = "bias" in module
            # Create a new instance of the custom linear layer
            new_layer = QUANT_LINEAR_MAPPING[quant_method](in_features, out_features, bias=bias, invert_weight_scales=True)

            # Replace the layer in the model
            if fuse_qkv and not any(name.endswith(suffix) for suffix in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                quantize_layers.append((name, new_layer))
            elif not fuse_qkv:
                quantize_layers.append((name, new_layer))
        if fuse_qkv and name not in modules_to_not_convert and module.__class__.__name__ == "Attention":
            # Replace Attention layers with BitLinearFusedAttention
            from mlx_lm.models.bitlinear_layers import BitLinearFusedAttention

            new_module = BitLinearFusedAttention(model.args)
            quantize_layers.append((name, new_module))
    if len(quantize_layers) > 0:
        model.update_modules(tree_unflatten(quantize_layers))
    return model

def bitnet_sanitze(model, weights):
    # Remove unused precomputed rotary freqs and handle QKV fusion
    sanitized = {}
    processed_layers = set()  # Track which layers we've already processed

    for k, v in weights.items():
        if "self_attn.rotary_emb.inv_freq" in k:
            continue

        if "self_attn" in k and ("o_proj" not in k and "attn_sub_norm" not in k):
            # Extract layer prefix
            prefix = k.split("self_attn")[0]

            # Only process each layer once
            if prefix not in processed_layers:
                processed_layers.add(prefix)

                # Handle QKV fusion for weights
                if f"{prefix}self_attn.q_proj.weight" in weights:
                    q = weights[f"{prefix}self_attn.q_proj.weight"]
                    k_weight = weights[f"{prefix}self_attn.k_proj.weight"]
                    v = weights[f"{prefix}self_attn.v_proj.weight"]
                    qkv = mx.concatenate([q, k_weight, v], axis=0)
                    sanitized[f"{prefix}self_attn.qkv_proj.weight"] = qkv

                # Handle weight scales if they exist
                if f"{prefix}self_attn.q_proj.weight_scale" in weights:
                    q_scale = weights[f"{prefix}self_attn.q_proj.weight_scale"]
                    k_scale = weights[f"{prefix}self_attn.k_proj.weight_scale"]
                    v_scale = weights[f"{prefix}self_attn.v_proj.weight_scale"]
                    # qkv_scale = mx.sqrt((q_scale**2 + k_scale**2 + v_scale**2) / 3) # Root mean square
                    sanitized[f"{prefix}self_attn.qkv_proj.weight_scale"] = mx.concatenate([q_scale, k_scale, v_scale], axis=0)

                # Handle biases if they exist
                if f"{prefix}self_attn.q_proj.bias" in weights:
                    q_bias = weights[f"{prefix}self_attn.q_proj.bias"]
                    k_bias = weights[f"{prefix}self_attn.k_proj.bias"]
                    v_bias = weights[f"{prefix}self_attn.v_proj.bias"]
                    qkv_bias = mx.concatenate([q_bias, k_bias, v_bias], axis=0)
                    sanitized[f"{prefix}self_attn.qkv_proj.bias"] = qkv_bias

            # Skip the individual q/k/v components since we've fused them
            continue
        else:
            sanitized[k] = v

    if model.args.tie_word_embeddings:
        sanitized.pop("lm_head.weight", None)
    return sanitized