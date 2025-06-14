import mlx.core as mx
import mlx.nn as nn


class BitLinear(nn.Module):
    """
    BitLinear module with memory-efficient weight handling.
    """
    def __init__(self, in_features, out_features, bias=True, dtype=mx.float16, invert_weight_scales = False, fuse_qkv = False):
        super().__init__()
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_qkv = fuse_qkv

        # Calculate packed dimensions - the first dimension gets packed 4:1
        # The weights are ternary so can be represented with 2 bits,
        # and they are packed in uint8 tensors, hence the number of values per item is 4
        packed_out_features = (out_features + 3) // 4
        self.weight = mx.zeros((packed_out_features, in_features), dtype=mx.uint8)

        self.invert_weight_scales = invert_weight_scales

        if fuse_qkv:
            self.weight_scale = mx.ones((3,), dtype=dtype)
        else:
            self.weight_scale = mx.array([1.0], dtype=dtype)

        if bias:
            self.bias = mx.zeros((out_features,), dtype=dtype)
        else:
            self.bias = None

        # Add kernel caches
        self._compiled_kernel = None
        self._compiled_qkv_kernel = None

    def bitlinear_kernel(self, x, packed_weights, out_features=None, scale=None):
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
        if (invert_weight_scales) {
            out[tid] = sum / weight_scale[0];
        } else {
            out[tid] = sum * weight_scale[0];
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

        out_features = out_features if self.fuse_qkv else self.out_features

        # Compile kernel once and cache it
        if self._compiled_kernel is None:
            self._compiled_kernel = mx.fast.metal_kernel(
                name="bitlinear_matmul",
                input_names=["x", "packed_weights", "weight_scale", "invert_weight_scales"],
                output_names=["out"],
                source=source,
            )

        outputs = self._compiled_kernel(
            inputs=[x_flattened.astype(self.dtype), packed_weights, scale, self.invert_weight_scales],
            template=[("batch_size", total_batch_elements), ("in_features", in_features), ("out_features", out_features)],
            grid=(total_batch_elements * out_features, 1, 1),
            threadgroup=(min(32, total_batch_elements * out_features), 1, 1),
            output_shapes=[(total_batch_elements, out_features)],
            output_dtypes=[self.dtype],
        )

        # Reshape output back to match input shape but with out_features as last dimension
        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (out_features,)
            return outputs[0].reshape(output_shape)
        else:
            return outputs[0]

    def bitlinear_fused_qkv_kernel(self, x, packed_weights, scales):
        """
        Custom Metal kernel that performs fused QKV computation in parallel.
        Handles Q (2560), K (640), V (640) outputs with their respective scales.
        """
        source = """
        uint tid = thread_position_in_grid.x;
        uint total_elements = batch_size * total_out_features;

        if (tid >= total_elements) return;

        uint batch_idx = tid / total_out_features;
        uint out_idx = tid % total_out_features;

        // Determine which component (Q, K, or V) this thread is computing
        uint component;
        uint local_out_idx;
        uint component_start_idx;
        uint component_out_features;
        uint weight_slice_start;

        if (out_idx < q_features) {
            // Q component
            component = 0;
            local_out_idx = out_idx;
            component_start_idx = 0;
            component_out_features = q_features;
            weight_slice_start = 0;
        } else if (out_idx < q_features + k_features) {
            // K component
            component = 1;
            local_out_idx = out_idx - q_features;
            component_start_idx = q_features;
            component_out_features = k_features;
            weight_slice_start = q_packed_rows;
        } else {
            // V component
            component = 2;
            local_out_idx = out_idx - q_features - k_features;
            component_start_idx = q_features + k_features;
            component_out_features = v_features;
            weight_slice_start = q_packed_rows + k_packed_rows;
        }

        float sum = 0.0;

        // Calculate packed dimensions for this component
        uint component_packed_rows = (component_out_features + 3) / 4;

        for (uint i = 0; i < in_features; i++) {
            // Get input value
            float x_val = x[batch_idx * in_features + i];

            // Determine which packed row and bit position within that packed value
            uint which_slice = local_out_idx / component_packed_rows;
            uint row_in_slice = local_out_idx % component_packed_rows;

            // Calculate the actual packed weight index for this component
            uint packed_idx = (weight_slice_start + row_in_slice) * in_features + i;
            uint8_t packed_val = packed_weights[packed_idx];

            // Extract the 2-bit value for this slice
            uint8_t mask = 3 << (2 * which_slice);  // 0b11 shifted to the right position
            uint8_t weight_bits = (packed_val & mask) >> (2 * which_slice);

            // Convert from {0,1,2} back to {-1,0,1}
            float weight_val = float(weight_bits) - 1.0;

            sum += x_val * weight_val;
        }

        // Apply component-specific weight scaling
        if (invert_weight_scales) {
            out[tid] = sum / scales[component];
        } else {
            out[tid] = sum * scales[component];
        }
        """

        # Handle multi-dimensional inputs by flattening all but the last dimension
        original_shape = x.shape
        if len(original_shape) > 2:
            x_flattened = x.reshape(-1, original_shape[-1])
            total_batch_elements = x_flattened.shape[0]
            in_features = x_flattened.shape[1]
        else:
            x_flattened = x
            total_batch_elements, in_features = x_flattened.shape

        # QKV dimensions (based on your split points [640, 800])
        q_features = 2560
        k_features = 640
        v_features = 640
        total_out_features = q_features + k_features + v_features

        # Calculate packed row counts for weight indexing
        q_packed_rows = (q_features + 3) // 4
        k_packed_rows = (k_features + 3) // 4

        # Compile kernel once and cache it
        if self._compiled_qkv_kernel is None:
            self._compiled_qkv_kernel = mx.fast.metal_kernel(
                name="bitlinear_fused_qkv",
                input_names=["x", "packed_weights", "scales", "invert_weight_scales"],
                output_names=["out"],
                source=source,
            )

        outputs = self._compiled_qkv_kernel(
            inputs=[x_flattened.astype(self.dtype), packed_weights, scales, self.invert_weight_scales],
            template=[
                ("batch_size", total_batch_elements),
                ("in_features", in_features),
                ("total_out_features", total_out_features),
                ("q_features", q_features),
                ("k_features", k_features),
                ("v_features", v_features),
                ("q_packed_rows", q_packed_rows),
                ("k_packed_rows", k_packed_rows)
            ],
            grid=(total_batch_elements * total_out_features, 1, 1),
            threadgroup=(min(32, total_batch_elements * total_out_features), 1, 1),
            output_shapes=[(total_batch_elements, total_out_features)],
            output_dtypes=[self.dtype],
        )

        # Reshape output back to match input shape but with total_out_features as last dimension
        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (total_out_features,)
            return outputs[0].reshape(output_shape)
        else:
            return outputs[0]

    def __call__(self, x):
        """
        Forward pass with weight scaling applied correctly.
        """
        org_dtype = x.dtype

        # Choose the appropriate kernel based on whether this is fused QKV
        if self.fuse_qkv:
            y = self.bitlinear_fused_qkv_kernel(x, self.weight, self.weight_scale)
        else:
            y = self.bitlinear_kernel(x, self.weight, scale=self.weight_scale)

        # Add bias if present
        if self.bias is not None:
            y = mx.add(y, self.bias)

        return y.astype(org_dtype)