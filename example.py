import torch
from kosmosg.main import KosmosG

# usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))

model = KosmosG()
output = model(img, text)
print(output)
