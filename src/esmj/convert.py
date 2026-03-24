"""Conversion framework: PyTorch ESM models → JAX/Equinox. Requires torch and esm."""

from dataclasses import fields
from functools import singledispatch

import equinox as eqx
import jax
import numpy as np
import torch
from jax import numpy as jnp

import esm
import esm.layers
import esm.layers.attention
import esm.layers.blocks
import esm.layers.transformer_stack
import esm.models
import esm.models.esmc

from esmj.model import (
    ESMC,
    Linear,
    LayerNorm,
    MultiHeadAttention,
    RotaryEmbedding,
    Sequential,
    SparseEmbedding,
    TransformerStack,
    UnifiedTransformerBlock,
    swiglu,
)


@singledispatch
def from_torch(x):
    raise NotImplementedError(f"from_torch not implemented for {type(x)}: {x}")


def convert_tensor(x: torch.Tensor):
    x = x.detach()
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
    return np.array(x)


# basic types
from_torch.register(torch.Tensor, convert_tensor)
from_torch.register(int, lambda x: x)
from_torch.register(float, lambda x: x)
from_torch.register(bool, lambda x: x)
from_torch.register(type(None), lambda x: x)
from_torch.register(tuple, lambda x: tuple(map(from_torch, x)))
from_torch.register(dict, lambda x: {k: from_torch(v) for k, v in x.items()})
from_torch.register(torch.nn.ReLU, lambda _: jax.nn.relu)
from_torch.register(torch.nn.GELU, lambda _: jax.nn.gelu)
from_torch.register(torch.nn.Sigmoid, lambda _: jax.nn.sigmoid)
from_torch.register(torch.nn.SiLU, lambda _: jax.nn.silu)
from_torch.register(torch.nn.ModuleList, lambda x: [from_torch(m) for m in x])


def _convert_by_fields(cls, model: torch.nn.Module):
    """Convert a torch module to an equinox module by matching dataclass fields."""
    field_to_type = {field.name: field.type for field in fields(cls)}
    kwargs = {
        child: from_torch(child_module)
        for child, child_module in model.named_children()
    } | {
        parameter_name: from_torch(parameter)
        for parameter_name, parameter in model.named_parameters(recurse=False)
    }

    for field_name, field_type in field_to_type.items():
        if not hasattr(model, field_name):
            if not isinstance(None, field_type):
                raise ValueError(
                    f"Field {field_name} for {cls} is not optional but is missing from torch model {model}"
                )
            else:
                kwargs[field_name] = None
        else:
            kwargs[field_name] = from_torch(getattr(model, field_name))

    torch_not_equinox = kwargs.keys() - field_to_type.keys()
    if torch_not_equinox:
        raise ValueError(
            f"Properties in torch model not found in equinox module {cls}: {torch_not_equinox}"
        )

    return cls(**kwargs)


# --- Layer registrations ---


@from_torch.register(torch.nn.Linear)
def _convert_linear(l: torch.nn.Linear):
    return Linear(weight=from_torch(l.weight), bias=from_torch(l.bias))


@from_torch.register(torch.nn.LayerNorm)
def _convert_layernorm(l: torch.nn.LayerNorm):
    return LayerNorm(
        weight=from_torch(l.weight), bias=from_torch(l.bias), eps=l.eps
    )


@from_torch.register(torch.nn.Sequential)
def _convert_sequential(module: torch.nn.Sequential):
    return Sequential(_modules=from_torch(module._modules))


@from_torch.register(torch.nn.modules.sparse.Embedding)
def _convert_embedding(m: torch.nn.modules.sparse.Embedding):
    return SparseEmbedding(embedding=eqx.nn.Embedding(weight=from_torch(m.weight)))


@from_torch.register(esm.layers.blocks.SwiGLU)
def _convert_swiglu(_):
    return swiglu


# --- ESM-specific registrations ---


@from_torch.register(esm.layers.rotary.RotaryEmbedding)
def _convert_rotary(m):
    return _convert_by_fields(RotaryEmbedding, m)


@from_torch.register(esm.layers.attention.MultiHeadAttention)
def _convert_mha(m):
    return _convert_by_fields(MultiHeadAttention, m)


@from_torch.register(esm.layers.blocks.UnifiedTransformerBlock)
def _convert_block(m):
    return _convert_by_fields(UnifiedTransformerBlock, m)


@from_torch.register(esm.layers.transformer_stack.TransformerStack)
def _convert_transformer_stack(m: esm.layers.transformer_stack.TransformerStack):
    blocks = [from_torch(b) for b in m.blocks]
    block_params = jax.tree.map(
        lambda *v: jnp.stack(v),
        *[eqx.filter(b, eqx.is_inexact_array) for b in blocks],
    )
    block_static = eqx.partition(blocks[0], eqx.is_inexact_array)[1]
    return TransformerStack(
        block_params=block_params,
        block_static=block_static,
        norm=from_torch(m.norm),
    )


@from_torch.register(esm.models.esmc.ESMC)
def _convert_esmc(m: esm.models.esmc.ESMC):
    return ESMC(
        embed=from_torch(m.embed),
        transformer=from_torch(m.transformer),
        sequence_head=from_torch(m.sequence_head),
        vocab=m.tokenizer.vocab
        | {esm.utils.constants.esm3.MASK_STR_SHORT: m.tokenizer.vocab["<mask>"]},
    )
