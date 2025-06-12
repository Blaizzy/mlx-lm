from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention
from .rope_utils import initialize_rope

class BitLinear(nn.Module):
    """
    BitLinear module with memory-efficient weight handling.
    """
    def __init__(self, in_features, out_features, bias=True, dtype=mx.float16, invert_weight_scales = False, fused_shapes = None):
        super().__init__()
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        # Calculate packed dimensions - the first dimension gets packed 4:1
        # The weights are ternary so can be represented with 2 bits,
        # and they are packed in uint8 tensors, hence the number of values per item is 4
        packed_out_features = (out_features + 3) // 4
        self.weight = mx.zeros((packed_out_features, in_features), dtype=mx.uint8)

        self.invert_weight_scales = invert_weight_scales
        self.fused_shapes = fused_shapes

        if fused_shapes is None:
            self.weight_scale = mx.array([1.0], dtype=dtype)
        else:
            self.weight_scale = mx.array([1.0] * (len(fused_shapes) + 1), dtype=dtype)

        self.fused_layers = fused_shapes is not None

        if bias:
            self.bias = mx.zeros((out_features,), dtype=dtype)
        else:
            self.bias = None

        # Add kernel cache
        self._compiled_kernel = None

    def bitlinear_kernel(self, x, packed_weights):
        """
        Custom Metal kernel that performs matrix multiplication directly on packed weights and scales the output.
        This eliminates the need to store unpacked weights in memory.
        """
        source = """
        uint tid = thread_position_in_grid.x;
        uint total_elements = batch_size * out_features;

        if (tid >= total_elements) return;

        uint batch_idx = tid / out_features;
        uint out_idx = tid % out_features;

        float sum = 0.0;

        // Calculate packed dimensions
        uint packed_rows = out_features / 4;  // Each packed row contains 4 output rows

        for (uint i = 0; i < in_features; i++) {
            // Get input value
            float x_val = x[batch_idx * in_features + i];

            // Determine which packed row and which bit position within that packed value
            uint which_slice = out_idx / packed_rows;  // Which of the 4 slices (0, 1, 2, 3)
            uint row_in_slice = out_idx % packed_rows;  // Which row within that slice

            // Get the packed weight value
            uint packed_idx = row_in_slice * in_features + i;
            uint8_t packed_val = packed_weights[packed_idx];

            // Extract the 2-bit value for this slice
            uint8_t mask = 3 << (2 * which_slice);  // 0b11 shifted to the right position
            uint8_t weight_bits = (packed_val & mask) >> (2 * which_slice);

            // Convert from {0,1,2} back to {-1,0,1}
            float weight_val = float(weight_bits) - 1.0;

            sum += x_val * weight_val;
        }

        // Apply weight scaling by diving them or multiplying them
        if (fused_layers) {
            out[tid] = sum;
        } else {
            if (invert_weight_scales) {
                out[tid] = sum / weight_scale[0];
            } else {
                out[tid] = sum * weight_scale[0];
            }
        }
        """

        # Handle multi-dimensional inputs by flattening all but the last dimension
        original_shape = x.shape
        if len(original_shape) > 2:
            # Flatten to (total_batch_elements, in_features)
            x_flattened = x.reshape(-1, original_shape[-1])
            total_batch_elements = x_flattened.shape[0]
            in_features = x_flattened.shape[1]
        else:
            x_flattened = x
            total_batch_elements, in_features = x_flattened.shape

        out_features = self.out_features

        # Compile kernel once and cache it
        if self._compiled_kernel is None:
            self._compiled_kernel = mx.fast.metal_kernel(
                name="bitlinear_matmul",
                input_names=["x", "packed_weights", "weight_scale", "invert_weight_scales", "fused_layers"],
                output_names=["out"],
                source=source,
            )

        outputs = self._compiled_kernel(
            inputs=[x_flattened.astype(self.dtype), packed_weights, self.weight_scale, self.invert_weight_scales, self.fused_layers],
            template=[("batch_size", total_batch_elements), ("in_features", in_features), ("out_features", out_features)],
            grid=(total_batch_elements * out_features, 1, 1),
            threadgroup=(min(256, total_batch_elements * out_features), 1, 1),
            output_shapes=[(total_batch_elements, out_features)],
            output_dtypes=[self.dtype],
        )
        
        if self.fused_layers:

            # First split the output into the number of fused shapes
            fused_outputs = mx.split(
                outputs[0], self.fused_shapes, axis=-1
            )
            # then apply the weight scaling to each output
            for i in range(len(fused_outputs)):
                if self.invert_weight_scales:
                    fused_outputs[i] = fused_outputs[i] / self.weight_scale[i]
                else:
                    fused_outputs[i] = fused_outputs[i] * self.weight_scale[i]
            
            # concat all outputs
            outputs[0] = mx.concatenate(fused_outputs, axis=-1)
            

        # Reshape output back to match input shape but with out_features as last dimension
        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (out_features,)
            return outputs[0].reshape(output_shape)
        else:
            return outputs[0]


    def __call__(self, x):
        """
        Forward pass with weight scaling applied correctly.
        """
        org_dtype = x.dtype

        # Use custom kernel for matrix multiplication directly on packed weights
        y = self.bitlinear_kernel(x, self.weight)

        # Add bias if present
        if self.bias is not None:
            y = mx.add(y, self.bias)

        return y.astype(org_dtype)
    

class BitLinearFusedAttention(nn.Module):
    def __init__(self, args):
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

        query_pos = n_heads * head_dim

        self.qkv_proj = BitLinear(
            dim,
            (n_heads + 2 * n_kv_heads) * head_dim,
            bias=attention_bias,
            fused_shapes=[query_pos, query_pos + self.n_kv_heads * self.head_dim],
        )
        self.o_proj = BitLinear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape
        
        qkv = self.qkv_proj(x)
        query_pos = self.n_heads * self.head_dim
        queries, keys, values = mx.split(
            qkv, [query_pos, query_pos + self.n_kv_heads * self.head_dim], axis=-1
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
        return self.o_proj(output)