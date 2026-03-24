"""JAX/Equinox model definitions for ESMC. No torch imports."""

import einops
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
import numpy as np


def swiglu(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jax.nn.silu(x1) * x2


class Linear(eqx.Module):
    """Linear layer that matches pytorch semantics."""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + jnp.broadcast_to(self.bias, x.shape[:-1] + (self.bias.shape[-1],))
        return o


class LayerNorm(eqx.Module):
    """LayerNorm that matches pytorch semantics."""

    weight: Float[Array, "Out"] | None
    bias: Float[Array, "Out"] | None
    eps: float

    def __call__(self, x: Float[Array, "... Out"]) -> Float[Array, "... Out"]:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        out = (x - mean) / jnp.sqrt(var + self.eps)
        if self.weight is not None:
            out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Sequential(eqx.Module):
    _modules: dict[str, eqx.Module]

    def __call__(self, x):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x


class SparseEmbedding(eqx.Module):
    embedding: eqx.nn.Embedding

    def __call__(self, indices):
        ndims = len(indices.shape)

        def apply(index):
            return self.embedding(index)

        f = apply
        for _ in range(ndims):
            f = jax.vmap(f)

        return f(indices)


class RotaryEmbedding(eqx.Module):
    dim: int
    base: int = 10000

    def __call__(self, q: Float[Array, "B N H D"], k: Float[Array, "B N H D"]):
        N = q.shape[1]
        t = jnp.arange(N, dtype=jnp.float32)
        freqs = jnp.outer(t, self.inverse_freq)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)

        return (
            self.apply_rotary_emb(q, cos, sin),
            self.apply_rotary_emb(k, cos, sin),
        )

    @property
    def inverse_freq(self):
        return 1.0 / (
            self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        )

    @staticmethod
    def rotate_half(x: Float[Array, "B N H D"]):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    @staticmethod
    def apply_rotary_emb(
        x: Float[Array, "B N H D"], cos: Float[Array, "N P"], sin: Float[Array, "N P"]
    ):
        ro_dim = cos.shape[-1] * 2
        assert ro_dim <= x.shape[-1]
        seqlen = x.shape[1]
        cos = cos[:seqlen]
        sin = sin[:seqlen]
        cos = einops.repeat(cos, "s d -> s 1 (2 d)")
        sin = einops.repeat(sin, "s d -> s 1 (2 d)")
        return jnp.concatenate(
            [
                x[..., :ro_dim] * cos
                + RotaryEmbedding.rotate_half(x[..., :ro_dim]) * sin,
                x[..., ro_dim:],
            ],
            axis=-1,
        )


class MultiHeadAttention(eqx.Module):
    d_model: int
    n_heads: int
    d_head: int
    layernorm_qkv: Sequential
    out_proj: Linear
    rotary: RotaryEmbedding
    q_ln: LayerNorm
    k_ln: LayerNorm

    def _apply_rotary(self, q, k):
        q = einops.rearrange(q, "... (h d) -> ... h d", h=self.n_heads, d=self.d_head)
        k = einops.rearrange(k, "... (h d) -> ... h d", h=self.n_heads, d=self.d_head)
        q, k = self.rotary(q, k)
        q = einops.rearrange(q, "... h d -> ... (h d)", h=self.n_heads, d=self.d_head)
        k = einops.rearrange(k, "... h d -> ... (h d)", h=self.n_heads, d=self.d_head)
        return q, k

    def __call__(self, x):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = jnp.split(qkv_BLD3, 3, axis=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD),
            self.k_ln(key_BLD),
        )

        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        query_BHLD, key_BHLD, value_BHLD = map(
            lambda x: einops.rearrange(
                x, pattern="b s (h d) -> b h s d", h=self.n_heads
            ),
            (query_BLD, key_BLD, value_BLD),
        )

        context_BHLD = jax.nn.dot_product_attention(
            einops.rearrange(query_BHLD, "B H S D -> B S H D"),
            einops.rearrange(key_BHLD, "B H S D -> B S H D"),
            einops.rearrange(value_BHLD, "B H S D -> B S H D"),
        )

        context_BHLD = einops.rearrange(context_BHLD, "b s h d -> b s (h d)")
        return self.out_proj(context_BHLD)


class UnifiedTransformerBlock(eqx.Module):
    ffn: Sequential
    attn: MultiHeadAttention
    scaling_factor: float

    def __call__(self, x):
        x = x + self.attn(x) / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x


class TransformerStack(eqx.Module):
    block_params: UnifiedTransformerBlock
    block_static: UnifiedTransformerBlock
    norm: LayerNorm

    def __call__(self, x: Float[Array, "B N D"]):
        def body(x, params):
            layer = eqx.combine(self.block_static, params)
            x = layer(x)
            return x, x

        final_state, all_states = jax.lax.scan(
            body,
            x,
            self.block_params,
        )
        return self.norm(final_state), all_states


class ESMCOutput(eqx.Module):
    logits: Float[Array, "B N V"]
    embedding: Float[Array, "B N D"]
    hiddens: Float[Array, "B N L D"]


class ESMC(eqx.Module):
    embed: SparseEmbedding
    transformer: TransformerStack
    sequence_head: Sequential
    vocab: dict[str, int]

    def __call__(self, tokens: Int[Array, "B N"]):
        assert tokens.ndim == 2, f"Expected 2D input, got {tokens.ndim}D"
        x = self.embed(tokens)
        x, hiddens = self.transformer(x)
        logits = self.sequence_head(x)
        return ESMCOutput(
            logits=logits,
            embedding=x,
            hiddens=hiddens,
        )

    def tokenize(self, sequence: str):
        return np.array(
            [self.vocab["<cls>"]]
            + [self.vocab[c] for c in sequence]
            + [self.vocab["<eos>"]],
            dtype=np.int32,
        )
