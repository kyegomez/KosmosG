[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# KosmosG
My implementation of the model KosmosG from "KOSMOS-G: Generating Images in Context with Multimodal Large Language Models"

## Installation
`pip install kosmosg`

## Usage
```python
import torch
from kosmosg.main import KosmosG

# usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))

model = KosmosG()
output = model(img, text)
print(output)
```

## Architecture
`text, image => KosmosG => text tokens with multi modality understanding`

# License
MIT

# Todo
- Create Aligner in pytorch
- Create Diffusion module
- Integrate these pieces
- Create a training script