import pytest
import torch
from kosmosg.main import KosmosG, ViTransformerWrapper, AutoregressiveWrapper

def test_kosmosg_initialization():
    model = KosmosG()
    assert isinstance(model, KosmosG)
    assert isinstance(model.encoder, ViTransformerWrapper)
    assert isinstance(model.decoder, AutoregressiveWrapper)

def test_kosmosg_forward():
    model = KosmosG()
    img = torch.randn(1, 3, 256, 256)
    text = torch.randint(0, 20000, (1, 1024))
    output = model(img, text)
    assert output.shape == (1, 1024, 20000)

@pytest.mark.parametrize("img_size, text_len", [(0, 1024), (256, 0), (0, 0)])
def test_kosmosg_forward_edge_cases(img_size, text_len):
    model = KosmosG()
    img = torch.randn(1, 3, img_size, img_size)
    text = torch.randint(0, 20000, (1, text_len))
    with pytest.raises(Exception):
        model(img, text)

def test_kosmosg_forward_invalid_input():
    model = KosmosG()
    img = torch.randn(1, 3, 256, 256)
    text = "invalid input"
    with pytest.raises(Exception):
        model(img, text)

def test_kosmosg_forward_invalid_dimensions():
    model = KosmosG()
    img = torch.randn(1, 3, 256)
    text = torch.randint(0, 20000, (1, 1024))
    with pytest.raises(Exception):
        model(img, text)