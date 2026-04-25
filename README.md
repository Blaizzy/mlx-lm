## MLX LM 

MLX LM is a Python package for generating text and fine-tuning large language
models on Apple silicon with MLX.

Some key features include:

* Integration with the Hugging Face Hub to easily use thousands of LLMs with a
  single command. 
* Support for quantizing and uploading models to the Hugging Face Hub.
* [Low-rank and full model
  fine-tuning](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
  with support for quantized models.
* Distributed inference and fine-tuning with `mx.distributed`

The easiest way to get started is to install the `mlx-lm` package:

**With `pip`**:

```sh
pip install mlx-lm
```

**With `conda`**:

```sh
conda install -c conda-forge mlx-lm
```

### Quick Start

To generate text with an LLM use:

```bash
mlx_lm.generate --prompt "How tall is Mt Everest?"
```

To chat with an LLM use:

```bash
mlx_lm.chat
```

This will give you a chat REPL that you can use to interact with the LLM. The
chat context is preserved during the lifetime of the REPL.

Commands in `mlx-lm` typically take command line options which let you specify
the model, sampling parameters, and more. Use `-h` to see a list of available
options for a command, e.g.:

```bash
mlx_lm.generate -h
```

The default model for generation and chat is
`mlx-community/Llama-3.2-3B-Instruct-4bit`.  You can specify any MLX-compatible
model with the `--model` flag. Thousands are available in the
[MLX Community](https://huggingface.co/mlx-community) Hugging Face
organization.

### Python API

You can use `mlx-lm` as a module:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

To see a description of all the arguments you can do:

```
>>> help(generate)
```

Check out the [generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/generate_response.py)
to see how to use the API in more detail. Check out the [batch generation
example](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/examples/batch_generate_response.py)
to see how to efficiently generate continuations for a batch of prompts.

The `mlx-lm` package also comes with functionality to quantize and optionally
upload models to the Hugging Face Hub.

You can convert models using the Python API:

```python
from mlx_lm import convert

repo = "mistralai/Mistral-7B-Instruct-v0.3"
upload_repo = "mlx-community/My-Mistral-7B-Instruct-v0.3-4bit"

convert(repo, quantize=True, upload_repo=upload_repo)
```

This will generate a 4-bit quantized Mistral 7B and upload it to the repo
`mlx-community/My-Mistral-7B-Instruct-v0.3-4bit`. It will also save the
converted model in the path `mlx_model` by default.

To see a description of all the arguments you can do:

```
>>> help(convert)
```

#### Streaming

For streaming generation, use the `stream_generate` function. This yields
a generation response object.

For example,

```python
from mlx_lm import load, stream_generate

repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(repo)

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)
print()
```

#### Sampling

The `generate` and `stream_generate` functions accept `sampler` and
`logits_processors` keyword arguments. A sampler is any callable which accepts
a possibly batched logits array and returns an array of sampled tokens.  The
`logits_processors` must be a list of callables which take the token history
and current logits as input and return the processed logits. The logits
processors are applied in order.

Some standard sampling functions and logits processors are provided in
`mlx_lm.sample_utils`.

### Command Line

You can also use `mlx-lm` from the command line with:

```
mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.3 --prompt "hello"
```

This will download a Mistral 7B model from the Hugging Face Hub and generate
text using the given prompt.

For a full list of options run:

```
mlx_lm.generate --help
```

To quantize a model from the command line run:

```
mlx_lm.convert --model mistralai/Mistral-7B-Instruct-v0.3 -q
```

For more options run:

```
mlx_lm.convert --help
```

You can upload new models to Hugging Face by specifying `--upload-repo` to
`convert`. For example, to upload a quantized Mistral-7B model to the
[MLX Hugging Face community](https://huggingface.co/mlx-community) you can do:

```
mlx_lm.convert \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    -q \
    --upload-repo mlx-community/my-4bit-mistral
```

Models can also be converted and quantized directly in the
[mlx-my-repo](https://huggingface.co/spaces/mlx-community/mlx-my-repo) Hugging
Face Space.

### Long Prompts and Generations 

`mlx-lm` has some tools to scale efficiently to long prompts and generations:

- A rotating fixed-size key-value cache.
- Prompt caching

To use the rotating key-value cache pass the argument `--max-kv-size n` where
`n` can be any integer. Smaller values like `512` will use very little RAM but
result in worse quality. Larger values like `4096` or higher will use more RAM
but have better quality.

Caching prompts can substantially speedup reusing the same long context with
different queries. To cache a prompt use `mlx_lm.cache_prompt`. For example:

```bash
cat prompt.txt | mlx_lm.cache_prompt \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --prompt - \
  --prompt-cache-file mistral_prompt.safetensors
``` 

Then use the cached prompt with `mlx_lm.generate`:

```
mlx_lm.generate \
    --prompt-cache-file mistral_prompt.safetensors \
    --prompt "\nSummarize the above text."
```

The cached prompt is treated as a prefix to the supplied prompt. Also notice
when using a cached prompt, the model to use is read from the cache and need
not be supplied explicitly.

Prompt caching can also be used in the Python API in order to avoid
recomputing the prompt. This is useful in multi-turn dialogues or across
requests that use the same context. See the
[example](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/chat.py)
for more usage details.

### Supported Models

`mlx-lm` supports thousands of LLMs available on the Hugging Face Hub. If the
model you want to run is not supported, file an
[issue](https://github.com/ml-explore/mlx-lm/issues/new) or better yet, submit
a pull request. Many supported models are available in various quantization
formats in the [MLX Community](https://huggingface.co/mlx-community) Hugging
Face organization.

For some models the tokenizer may require you to enable the `trust_remote_code`
option. You can do this by passing `--trust-remote-code` in the command line.
If you don't specify the flag explicitly, you will be prompted to trust remote
code in the terminal when running the model. 

Tokenizer options can also be set in the Python API. For example:

```python
model, tokenizer = load(
    "qwen/Qwen-7B",
    tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
)
```

### DeepSeek V4 / DeepSeek V4 Flash

This fork adds native support for **DeepSeek-V4** and **DeepSeek-V4-Flash** on Apple Silicon, including full Metal kernel acceleration.

#### Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("path/to/deepseek-v4-flash")
text = generate(model, tokenizer, prompt="Explain attention sinks.", verbose=True)
```

Or from the command line:

```bash
mlx_lm.generate --model path/to/deepseek-v4-flash --prompt "Explain attention sinks."
```

The model type is `deepseek_v4`. Pre-quantized checkpoints (FP8 or FP4 experts) are loaded and dequantized automatically — no manual conversion step is required.

#### Architecture

DeepSeek V4 Flash introduces several architectural innovations that this implementation fully supports:

**HyperConnection** replaces standard residual connections. Each layer maintains `hc_mult=4` parallel hidden streams that are combined through a learned Sinkhorn-normalized mixing matrix. A custom Metal kernel (`_make_hc_split_sinkhorn_kernel`) computes the 4×4 doubly-stochastic combination weights using float4 SIMD and online Sinkhorn iterations entirely on-GPU.

**Compressed attention (Compressor + Indexer)** provides long-range context without quadratic cost. At every layer with `compress_ratio > 0`, hidden states are pooled into a compressed KV sequence (at ratio 4 with overlap or ratio 128 for long range). During decode, an x-buffer defers the expensive `wkv`/`wgate` projections until a full compression window is ready, saving `(ratio−1)/ratio` GEMVs per step. An Indexer then selects the top-k most relevant compressed KV entries per query head using learned index projections.

**Sparse attention paths** handle prefill and decode differently:
- Prefill (`L > 1`): a fused Metal kernel (`ds4_fused_sparse_attn`) computes online softmax over the local sliding-window KV and the top-k sparse compressed KV in a single pass, avoiding materialising the `[B, L, topk, D]` gather intermediate.
- Decode (`L = 1`): the indexer is skipped (compressed pool fits within `index_topk` anyway); standard SDPA is used over `[local KV ∥ pooled KV]`.

**Attention sinks** add a learnable virtual token to every attention layer whose score is a per-head bias and whose value contribution is zero, stabilising attention distributions over long contexts.

**Mixture of Experts** uses 256 routed experts plus 1 shared expert per MoE layer. Expert routing uses `sqrtsoftplus` scoring with auxiliary-free top-k selection (`noaux_tc`). The first `num_hash_layers` layers use hash-based routing. `LimitedSwiGLU` clamps gate and up projections to prevent activation overflow.

**Grouped output projection** splits the large O-projection into 8 groups (`o_groups=8`), each with its own low-rank A matrix (`wo_a`) shared across groups, reducing peak memory during the projection step.

**Partial RoPE** applies rotary embeddings only to the `qk_rope_head_dim`-sized suffix of each head, leaving the `nope` prefix unrotated. A dedicated Metal kernel (`ds4_partial_rope`) fuses this with the split/concat that a naive implementation would require.

**Per-head query RMS norm** (`ds4_q_norm`) normalises each query head in a fused Metal kernel before the RoPE step, replacing the `mx.rsqrt` + elementwise pattern.

#### Metal Kernels

| Kernel | Purpose |
|---|---|
| `ds4_partial_rope` | Fused partial RoPE — eliminates split/concat intermediate |
| `ds4_q_norm` | Per-head query RMS normalisation |
| `_make_hc_split_sinkhorn_kernel` | float4 SIMD Sinkhorn for HyperConnection mixing weights |
| `ds4_fused_sparse_attn` | Online-softmax prefill over local + sparse KV + attention sink |
| `_split_sparse_attention` | MLX fallback for `ds4_fused_sparse_attn` on CPU or older Metal |

All kernels fall back gracefully to pure MLX when Metal is unavailable.

#### Quantization Support

The `sanitize` method handles pre-quantized checkpoints transparently:

- **FP8 weights** (E4M3/E5M2, block-scaled) are dequantized to BF16 on load.
- **FP4/MXFP4 expert weights** are unpacked from the 4-bit lookup table and dequantized to BF16, then re-quantized using MLX's native group-quantized matmul format.
- The safetensors loader is extended to reinterpret the `F8_E8M0` dtype used by some HuggingFace checkpoints that standard MLX cannot parse.

Precision-sensitive parameters (attention sinks, HyperConnection base/scale, expert correction biases) are excluded from any subsequent `cast` operations via `cast_predicate`.

#### Cache

`DeepseekV4Cache` wraps `RotatingKVCache` for the local sliding window and adds two parallel state buffers (compressor and indexer). It implements the full `BatchRotatingKVCache` interface — supporting `extract`, `extend`, `merge`, `filter`, and `trim` — so batch generation and prompt caching work out of the box.

#### Infrastructure Changes

- **`tokenizer_utils.py`**: adds a fallback to `PreTrainedTokenizerFast` when `AutoTokenizer` raises `AttributeError` on custom model types whose config triggers transformers' `rope_scaling` standardisation before `max_position_embeddings` is available.
- **`utils.py`**: adds `_load_safetensors` which patches `F8_E8M0` dtype headers in-place before loading, allowing FP8-quantized checkpoints to be loaded without a separate conversion step.

### Large Models

> [!NOTE]
    This requires macOS 15.0 or higher to work.

Models which are large relative to the total RAM available on the machine can
be slow. `mlx-lm` will attempt to make them faster by wiring the memory
occupied by the model and cache. This requires macOS 15 or higher to
work.

If you see the following warning message:

> [WARNING] Generating with a model that requires ...

then the model will likely be slow on the given machine. If the model fits in
RAM then it can often be sped up by increasing the system wired memory limit.
To increase the limit, set the following `sysctl`:

```bash
sudo sysctl iogpu.wired_limit_mb=N
```

The value `N` should be larger than the size of the model in megabytes but
smaller than the memory size of the machine.
