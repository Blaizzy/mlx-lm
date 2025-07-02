# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .bitlinear_layers import BitLinear
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads



class BitFusedAttention(nn.Module):
    def __init__(self, args: ModelArgs, invert_weight_scales: bool = False, add_sub_norm: bool = True):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.qkv_proj = BitLinear(
            dim,
            (n_heads + 2 * n_kv_heads) * head_dim,
            bias=attention_bias,
            fused_qkv=True,
            invert_weight_scales=invert_weight_scales,
        )
        self.o_proj = BitLinear(n_heads * head_dim, dim, bias=attention_bias, invert_weight_scales=invert_weight_scales)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )
        self.add_sub_norm = add_sub_norm
        if self.add_sub_norm:
            self.attn_sub_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Fused QKV projection
        query_pos = self.n_heads * self.head_dim
        kv_pos = self.n_kv_heads * self.head_dim

        kwargs = {
            "query_position": query_pos,
            "kv_position": kv_pos,
        }
        qkv = self.qkv_proj(x, **kwargs)

        queries, keys, values = mx.split(
            qkv, [query_pos, query_pos + kv_pos], axis=-1
        )

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.add_sub_norm:
            output = self.attn_sub_norm(output)
        output = self.o_proj(output)

        return output


@partial(mx.compile, shapeless=True)
def relu2(x):
    return mx.square(nn.relu(x))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.gate_proj = BitLinear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = BitLinear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = BitLinear(dim, hidden_dim, bias=mlp_bias)
        self.ffn_sub_norm = nn.RMSNorm(args.intermediate_size, eps=args.rms_norm_eps)

    def __call__(self, x) -> mx.array:
        x = relu2(self.gate_proj(x)) * self.up_proj(x)
        x = self.ffn_sub_norm(x)
        x = self.down_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = BitFusedAttention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs and handle QKV fusion
        sanitized_weights = {}
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
                        k = weights[f"{prefix}self_attn.k_proj.weight"]
                        v = weights[f"{prefix}self_attn.v_proj.weight"]
                        qkv = mx.concatenate([q, k, v], axis=0)
                        sanitized_weights[f"{prefix}self_attn.qkv_proj.weight"] = qkv

                    # Handle weight scales if they exist
                    if f"{prefix}self_attn.q_proj.weight_scale" in weights:
                        q_scale = weights[f"{prefix}self_attn.q_proj.weight_scale"]
                        k_scale = weights[f"{prefix}self_attn.k_proj.weight_scale"]
                        v_scale = weights[f"{prefix}self_attn.v_proj.weight_scale"]

                        sanitized_weights[f"{prefix}self_attn.qkv_proj.weight_scale"] = mx.concatenate([q_scale, k_scale, v_scale], axis=0)


                    # Handle biases if they exist
                    if f"{prefix}self_attn.q_proj.bias" in weights:
                        q_bias = weights[f"{prefix}self_attn.q_proj.bias"]
                        k_bias = weights[f"{prefix}self_attn.k_proj.bias"]
                        v_bias = weights[f"{prefix}self_attn.v_proj.bias"]
                        qkv_bias = mx.concatenate([q_bias, k_bias, v_bias], axis=0)
                        sanitized_weights[f"{prefix}self_attn.qkv_proj.bias"] = qkv_bias

                # Skip the individual q/k/v components since we've fused them
                continue
            else:
                sanitized_weights[k] = v

        if self.args.tie_word_embeddings:
            sanitized_weights.pop("lm_head.weight", None)
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers
