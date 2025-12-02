# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from . import llama
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    def __post_init__(self):
        if "tie_word_embeddings" not in self.text_config:
            self.text_config["tie_word_embeddings"] = False


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = llama.Model(llama.ModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        def dequant(weight, scale_inv):
            dtype = weight.dtype
            if scale_inv.ndim < 2:
                # Per-tensor dequantization
                return (weight * scale_inv).astype(dtype)
            else:
                # Per-block dequantization
                bs = 128  # block size
                m, n = weight.shape
                pad_bottom = (-m) % bs
                pad_side = (-n) % bs
                weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
                weight = weight.reshape(
                    ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
                )
                weight = (weight * scale_inv[:, None, :, None]).reshape(
                    m + pad_bottom, n + pad_side
                )
                return weight[:m, :n].astype(dtype)

        # Remap for int4
        new_weights = {}
        for k, v in weights.items():
            if k.endswith("weight_shape"):
                base = k.replace("weight_shape", "")
                new_weights[base + "weight"] = weights[base + "weight_packed"].view(
                    mx.uint32
                )
                s = weights[base + "weight_scale"]
                new_weights[base + "scales"] = s
                new_weights[base + "biases"] = -8 * s
            elif not (k.endswith("weight_scale") or k.endswith("weight_packed")):
                new_weights[k] = v
        weights = new_weights

        # Dequantize fp8
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                weight = dequant(weight, scale_inv)
                new_weights[wk] = weight
            elif "activation_scale" in k:
                continue
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith(("vision_tower", "multi_modal_projector"))
        }

        return weights

    @property
    def layers(self):
        return self.language_model.model.layers
