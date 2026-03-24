"""Download and load pretrained models from HuggingFace Hub."""

import os

from esmj.serialization import load_model

HF_REPO = "escalante-bio/esmj"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "esmj", "weights")
_MODEL_NAMES = ("esmc_300m",)


def ensure_weights(cache_dir: str = DEFAULT_CACHE, models: tuple[str, ...] = _MODEL_NAMES):
    """Download model weights from HuggingFace if not already cached."""
    from huggingface_hub import hf_hub_download

    for name in models:
        for ext in (".eqx", ".skeleton.pkl"):
            path = os.path.join(cache_dir, f"{name}{ext}")
            if not os.path.exists(path):
                os.makedirs(cache_dir, exist_ok=True)
                hf_hub_download(HF_REPO, f"{name}{ext}", local_dir=cache_dir)


def from_pretrained(model_name: str = "esmc_300m", cache_dir: str = DEFAULT_CACHE):
    """Load a pretrained ESMC model, downloading weights if needed."""
    ensure_weights(cache_dir, models=(model_name,))
    return load_model(os.path.join(cache_dir, model_name))
