
import pytest
from tokenizer import Tokenizer, CM3LeonTokenizer
from sentencepiece import SentencePieceProcessor
from transformers import CLIPModel, CLIPProcessor


def test_tokenizer_initialization():
    model_path = "path/to/model"
    tokenizer = Tokenizer(model_path)
    assert isinstance(tokenizer, Tokenizer)
    assert isinstance(tokenizer.sp_model, SentencePieceProcessor)
    assert tokenizer.n_words == tokenizer.sp_model.vocab_size()
    assert tokenizer.bos_id == tokenizer.sp_model.bos_id()
    assert tokenizer.eos_id == tokenizer.sp_model.eos_id()
    assert tokenizer.pad_id == tokenizer.sp_model.pad_id()

def test_tokenizer_encode_decode():
    model_path = "path/to/model"
    tokenizer = Tokenizer(model_path)
    text = "this is a test"
    encoded = tokenizer.encode(text, bos=True, eos=True)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

def test_cm3leontokenizer_initialization():
    model_path = "path/to/model"
    tokenizer = CM3LeonTokenizer(model_path)
    assert isinstance(tokenizer, CM3LeonTokenizer)
    assert isinstance(tokenizer.processor, CLIPProcessor)
    assert isinstance(tokenizer.model, CLIPModel)

def test_cm3leontokenizer_encoded():
    model_path = "path/to/model"
    tokenizer = CM3LeonTokenizer(model_path)
    text = "this is a test"
    image_path = "path/to/image.jpg"
    encoded = tokenizer.encoded(s=text, image=image_path)
    assert isinstance(encoded, list)

@pytest.mark.parametrize("text, image_path", [(None, "path/to/image.jpg"), ("this is a test", None), (None, None)])
def test_cm3leontokenizer_encoded_edge_cases(text, image_path):
    model_path = "path/to/model"
    tokenizer = CM3LeonTokenizer(model_path)
    with pytest.raises(Exception):
        tokenizer.encoded(s=text, image=image_path)

def test_cm3leontokenizer_encoded_invalid_image():
    model_path = "path/to/model"
    tokenizer = CM3LeonTokenizer(model_path)
    text = "this is a test"
    image_path = "path/to/invalid_image.jpg"
    with pytest.raises(Exception):
        tokenizer.encoded(s=text, image=image_path)