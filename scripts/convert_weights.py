"""Convert PyTorch ESMC checkpoints to torch-free .eqx files."""

import os

from esm.models.esmc import ESMC

from esmj.convert import from_torch
from esmj.serialization import save_model

os.makedirs("weights", exist_ok=True)

print("Loading esmc_300m...")
client = ESMC.from_pretrained("esmc_300m").to("cpu")

print("Converting to JAX...")
jax_model = from_torch(client)

print("Saving weights...")
save_model(jax_model, "weights/esmc_300m")
print("Saved weights/esmc_300m.eqx + weights/esmc_300m.skeleton.pkl")
