import mlx.core as mx
import mlx.nn as nn


class BitLinear(nn.Module):
    """
    BitLinear module with memory-efficient weight handling.
    """
    def __init__(self, in_features, out_features, *, bias=True, dtype=mx.float16,
                 invert_weight_scales: bool = False, fused_qkv: bool = False):
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
        self.fused_qkv = fused_qkv

        shape = [1.0, 1.0, 1.0] if fused_qkv else [1.0]
        self.weight_scale = mx.array(shape, dtype=dtype)

        if bias:
            self.bias = mx.zeros((out_features,), dtype=dtype)
        else:
            self.bias = None

        self._compiled_kernel = self._compile_qkv_kernel() if fused_qkv else self._compile_matmul_kernel()


    def _compile_matmul_kernel(self):
        """
        Custom Metal kernel that performs matrix multiplication directly on packed weights and scales the output.
        This eliminates the need to store unpacked weights in memory.
        """
        source = r"""
        using namespace metal;
        uint tid = thread_position_in_grid.x;
        if (tid >= batch_size * out_features) return;

        uint batch_idx = tid / out_features;
        uint out_idx   = tid % out_features;

        // Calculate packed dimensions
        uint packed_rows   = (out_features + 3) / 4;
        uint which_slice   = out_idx / packed_rows;
        uint row_in_slice  = out_idx - which_slice * packed_rows;
        uint row_offset    = row_in_slice * in_features;
        uint batch_offset  = batch_idx * in_features;
        uint shift_bits    = 2u * which_slice;

        float sum = 0.0f;
        uint i = 0u;
        // ------------ 8‑feature unrolled loop (2 × float4) ------------
        for (; i + 7u < in_features; i += 8u) {
            // Get input values
            float4 x0 = float4(x[batch_offset + i + 0u], x[batch_offset + i + 1u],
                              x[batch_offset + i + 2u], x[batch_offset + i + 3u]);
            float4 x1 = float4(x[batch_offset + i + 4u], x[batch_offset + i + 5u],
                              x[batch_offset + i + 6u], x[batch_offset + i + 7u]);

            // Get weights
            uchar4 pv0 = *reinterpret_cast<const device uchar4*>(packed_weights + row_offset + i);
            uchar4 pv1 = *reinterpret_cast<const device uchar4*>(packed_weights + row_offset + i + 4u);

            // Extract the 2-bit slice; {0,1,2} -> {-1,0,1} (11 is unused and would map to 2)
            uint4  u0  = uint4(pv0) >> shift_bits;
            uint4  u1  = uint4(pv1) >> shift_bits;
            float4 w0 = float4(u0 & uint4(3u)) - 1.0f;
            float4 w1 = float4(u1 & uint4(3u)) - 1.0f;

            sum += dot(x0, w0) + dot(x1, w1);
        }


        // Apply weight scaling by diving them or multiplying them
        out[tid] = invert_weight_scales ? (sum / weight_scale[0])
                                        : (sum * weight_scale[0]);
        """
        return mx.fast.metal_kernel(
            name="bitlinear_matmul_8unroll",
            input_names=["x", "packed_weights", "weight_scale", "invert_weight_scales"],
            output_names=["out"],
            source=source,
        )

    def _compile_qkv_kernel(self):
        """
        Custom Metal kernel that performs fused QKV computation in parallel.
        Handles Q, K, V outputs with their respective scales.
        """
        source = r"""
        using namespace metal;
        uint tid = thread_position_in_grid.x;
        if (tid >= batch_size * total_out_features) return;

        uint batch_idx = tid / total_out_features;
        uint out_idx   = tid % total_out_features;

        // Determine component once per output (Q, K, V)
        uint component, local_out_idx, weight_slice_start, comp_out_feat;
        if (out_idx < q_features) {
            component = 0u; local_out_idx = out_idx; comp_out_feat = q_features; weight_slice_start = 0u;
        } else if (out_idx < q_features + k_features) {
            component = 1u; local_out_idx = out_idx - q_features; comp_out_feat = k_features; weight_slice_start = q_packed_rows;
        } else {
            component = 2u; local_out_idx = out_idx - q_features - k_features; comp_out_feat = v_features; weight_slice_start = q_packed_rows + k_packed_rows;
        }

        // Calculate packed dimensions
        uint comp_packed_rows = (comp_out_feat + 3) / 4;
        uint which_slice      = local_out_idx / comp_packed_rows;
        uint row_in_slice     = local_out_idx - which_slice * comp_packed_rows;
        uint row_offset       = (weight_slice_start + row_in_slice) * in_features;
        uint batch_offset     = batch_idx * in_features;
        uint shift_bits       = 2u * which_slice;

        float sum = 0.0f;
        uint i = 0u;
        // ------------ 8‑feature unrolled loop (2 × float4) ------------
        for (; i + 7u < in_features; i += 8u) {
            // Get input values
            float4 x0 = float4(x[batch_offset + i + 0u], x[batch_offset + i + 1u],
                              x[batch_offset + i + 2u], x[batch_offset + i + 3u]);
            float4 x1 = float4(x[batch_offset + i + 4u], x[batch_offset + i + 5u],
                              x[batch_offset + i + 6u], x[batch_offset + i + 7u]);

            // Get weights
            uchar4 pv0 = *reinterpret_cast<const device uchar4*>(packed_weights + row_offset + i);
            uchar4 pv1 = *reinterpret_cast<const device uchar4*>(packed_weights + row_offset + i + 4u);

            // Extract the 2-bit slice; {0,1,2} -> {-1,0,1} (11 is unused and would map to 2)
            uint4  u0  = uint4(pv0) >> shift_bits;
            uint4  u1  = uint4(pv1) >> shift_bits;
            float4 w0 = float4(u0 & uint4(3u)) - 1.0f;
            float4 w1 = float4(u1 & uint4(3u)) - 1.0f;

            sum += dot(x0, w0) + dot(x1, w1);
        }

        // Apply weight scaling by diving them or multiplying them
        out[tid] = invert_weight_scales ? (sum / scales[component])
                                        : (sum * scales[component]);
        """
        return mx.fast.metal_kernel(
            name="bitlinear_fused_qkv_8unroll",
            input_names=["x", "packed_weights", "scales", "invert_weight_scales"],
            output_names=["out"],
            source=source,
        )


    def _flatten_input(self, x: mx.array):
        if x.ndim <= 2:
            return x, x.shape[0], x.shape[1]
        flat = x.reshape(-1, x.shape[-1])
        return flat, flat.shape[0], flat.shape[1]

    def _dispatch_qkv(self, x: mx.array, packed_weights: mx.array, *,
                      query_position: int, kv_position: int):
        shape = x.shape
        x_flat, batches, in_feat = self._flatten_input(x)
        q = query_position; k = v = kv_position
        total = q + k + v
        q_pack = (q + 3) // 4; k_pack = (k + 3) // 4

        out = self._compiled_kernel(
            inputs=[x_flat.astype(self.dtype), packed_weights, self.weight_scale, self.invert_weight_scales],
            template=[
                ("batch_size", batches), ("in_features", in_feat),
                ("total_out_features", total), ("q_features", q),
                ("k_features", k), ("v_features", v),
                ("q_packed_rows", q_pack), ("k_packed_rows", k_pack),
            ],
            grid=(batches * total, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(batches, total)], output_dtypes=[self.dtype],
        )[0]
        return out.reshape(shape[:-1] + (total,)) if x.ndim > 2 else out

    def _dispatch_matmul(self, x: mx.array, packed_weights: mx.array):
        shape = x.shape
        x_flat, batches, in_feat = self._flatten_input(x)
        out = self._compiled_kernel(
            inputs=[x_flat.astype(self.dtype), packed_weights, self.weight_scale, self.invert_weight_scales],
            template=[("batch_size", batches), ("in_features", in_feat),
                      ("out_features", self.out_features)],
            grid=(batches * self.out_features, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(batches, self.out_features)], output_dtypes=[self.dtype],
        )[0]
        return out.reshape(shape[:-1] + (self.out_features,)) if x.ndim > 2 else out


    def __call__(self, x: mx.array, **kwargs):
        out = (self._dispatch_qkv(x, self.weight, **kwargs) if self.fused_qkv
               else self._dispatch_matmul(x, self.weight))
        if self.bias is not None:
            out = mx.add(out, self.bias)
        return out.astype(x.dtype)


    @staticmethod
    def benchmark():
        import time
        cases = [
            ("Tiny prompt", 1, 5, 4096), ("Small prompt", 1, 11, 4096),
            ("Medium prompt", 1, 32, 4096), ("Large prompt", 1, 128, 4096),
            ("Generation", 1, 200, 4096), ("Batch generation", 8, 100, 4096),
        ]
        for name, bs, sl, hs in cases:
            model = BitLinear(hs, hs, fused_qkv=False)
            x = mx.random.normal((bs, sl, hs))
            for _ in range(5): model(x)
            t0 = time.time()
            for _ in range(100): mx.eval(model(x))
            dt = (time.time() - t0) / 100
            print(f"{name:<16}: {dt*1e3:.1f} ms | {(bs*sl)/dt:,.0f} tok/s")


if __name__ == "__main__":
    BitLinear.benchmark()
