"""JAX translation of ESMC."""

from esmj.model import (
    ESMC,
    ESMCOutput,
    Linear,
    LayerNorm,
    MultiHeadAttention,
    RotaryEmbedding,
    Sequential,
    SparseEmbedding,
    TransformerStack,
    UnifiedTransformerBlock,
)
from esmj.hub import from_pretrained
from esmj.serialization import save_model, load_model


def from_torch(x):
    """Convert a PyTorch ESM model to JAX/Equinox. Requires torch and esm."""
    from esmj.convert import from_torch as _from_torch

    return _from_torch(x)
