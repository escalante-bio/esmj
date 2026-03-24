Barebones translation of [ESMC](https://github.com/evolutionaryscale/esm) to JAX/[equinox](https://docs.kidger.site/equinox/).

## Installation

```bash
# inference only (no torch needed)
uv pip install esmj

# with conversion support (installs torch + esm)
uv pip install esmj[convert]
```

## Usage: Pretrained model (no torch)

```python
from esmj import from_pretrained
import equinox as eqx

# downloads weights from HuggingFace on first call
model = from_pretrained("esmc_300m")
tokens = model.tokenize("ESCALANTE")

model = eqx.filter_jit(model)
output = model(tokens[None])  # add batch dimension

output.logits    # (B, N, V)
output.embedding # (B, N, D)
output.hiddens   # (B, N, L, D)
```

## Usage: Convert from PyTorch

```python
from esmj import from_torch, save_model
import equinox as eqx
import numpy as np

# load torch model
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
client = ESMC.from_pretrained("esmc_300m").to("cpu")

# torch prediction
prot_seq = "ESCALANTE"
protein = ESMProtein(sequence=prot_seq)
protein_tensor = client.encode(protein)
torch_output = client.logits(
   protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)

# translate model to JAX
eqx_model = from_torch(client)
tokens = eqx_model.tokenize(prot_seq)
eqx_model = eqx.filter_jit(eqx_model)
output = eqx_model(tokens[None])

print(np.abs(output.logits - np.array(torch_output.logits.sequence)).max())

# save for later use without torch
save_model(eqx_model, "esmc_300m")
```
