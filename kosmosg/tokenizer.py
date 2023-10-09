import os
from logging import getLogger
from typing import List, Optional, Union
from PIL import Image
from sentencepiece import SentencePieceProcessor
from transformers import CLIPProcessor, CLIPModel

logger = getLogger()


class Tokenizer:
    """
    A SentencePieceTokenizer is a tokenizer that uses a pretrained SentencePiece model
    to convert text into tokens and vice versa.

    Parameters:
    - model_path (str): Path to the pretrained SentencePiece model file.

    Attributes:
    - n_words (int): Vocabulary size of the SentencePiece model.
    - bos_id (int): Token ID of the beginning-of-sentence (BOS) token.
    - eos_id (int): Token ID of the end-of-sentence (EOS) token.
    - pad_id (int): Token ID of the padding (PAD) token.
    """

    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos is True:
            t = [self.bos_id] + t
        if eos is True:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


class CM3LeonTokenizer(Tokenizer):
    """
    CM3LeonTokenizer is a tokenizer that uses a pretrained SentencePiece model
    to convert text into tokens and vice versa.

    Parameters:
    - model_path (str): Path to the pretrained SentencePiece model file.

    Usage
    -----
    tokenizer = CM3LeonTokenizer(model_path="path/to/model")
    tokens = tokenizer.encode("this is a description")
    print(tokens)
    """
    def __init__(
        self,
        model_path: str
    ):
        super().__init__(model_path)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


    def encoded(
        self,
        s: str = None,
        image: str = None,
    ) -> List[Union[int, float]]:
        """Encodes the image and text into a sequence of tokens and embeddings"""

        #encode text
        text = self.encode(
            s,
            bos=True,
            eos=False,
        )

        # Load the image
        image = Image.open(image)

        #get embeds for img
        inputs = self.processor(text=s, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)

        #combine text tokens and image embeddings
        seq = text + outputs.image_embeds.tolist()
        return seq
    

tokenizer = CM3LeonTokenizer(model_path="tokenizers/sentencepiece.bpe.model")
tokens = tokenizer.encoded("this is a description", image="agorabanner.png")
print(tokens)