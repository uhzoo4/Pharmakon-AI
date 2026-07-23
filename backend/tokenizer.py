from __future__ import annotations

from typing import List, Sequence, Union, Tuple
import tiktoken

class Tokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding_name: str = encoding_name
        self.enc: tiktoken.Encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size: int = getattr(self.enc, "n_vocab", 50257)

    def encode(self, text: str) -> List[int]:
        assert isinstance(text, str), f"Expected input to be str, got {type(text)}"
        tokens = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        assert isinstance(tokens, list), f"Expected returned tokens to be list, got {type(tokens)}"
        return tokens

    def decode(self, tokens: Union[Sequence[int], List[int], Tuple[int, ...]]) -> str:
        assert isinstance(tokens, (list, tuple)), f"Expected tokens to be list or tuple, got {type(tokens)}"
        text = self.enc.decode(list(tokens))
        assert isinstance(text, str), f"Expected decoded output to be str, got {type(text)}"
        return text

    def test_identity(self, sample_text: str) -> bool:
        """
        Runs encode -> decode identity test on a held-out text sample.
        Raises AssertionError if the decoded text does not match the original.
        """
        encoded = self.encode(sample_text)
        decoded = self.decode(encoded)
        assert decoded == sample_text, (
            f"Tokenizer identity test failed!\n"
            f"Original: {repr(sample_text)}\n"
            f"Decoded:  {repr(decoded)}"
        )
        return True
