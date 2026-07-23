"""
Pharmakon Tokenizer Wrapper
Wraps `tiktoken` for reliable tokenization with identity verification tests.
"""

import tiktoken
from typing import List

class Tokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        assert isinstance(text, str), f"Expected input to be str, got {type(text)}"
        tokens = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        assert isinstance(tokens, list), f"Expected returned tokens to be list, got {type(tokens)}"
        return tokens

    def decode(self, tokens: List[int]) -> str:
        assert isinstance(tokens, (list, tuple)), f"Expected tokens to be list or tuple, got {type(tokens)}"
        text = self.enc.decode(tokens)
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
