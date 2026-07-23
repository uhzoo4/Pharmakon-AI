"""
Pharmakon Dataset & Memmap Sampler Module
Provides binary memory-mapped dataset creation and batch sampling with explicit boundary shape assertions.
"""

import os
import numpy as np
from typing import Tuple, List, Generator, Union, Any

def create_memmap_bin(
    tokens: Union[List[int], np.ndarray],
    bin_path: str,
    dtype: Any = np.uint16
) -> int:
    """
    Writes a list/array of token IDs into a memory-mapped binary file.
    Returns total token count.
    """
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    arr = np.array(tokens, dtype=dtype)
    num_tokens = len(arr)
    assert num_tokens > 0, "Cannot write empty token array to memmap bin"
    
    fp = np.memmap(bin_path, dtype=dtype, mode='w+', shape=(num_tokens,))
    fp[:] = arr[:]
    fp.flush()
    del fp
    return num_tokens

class MemmapBatchSampler:
    def __init__(
        self,
        bin_path: str,
        seq_len: int = 64,
        batch_size: int = 16,
        dtype: Any = np.uint16
    ):
        assert os.path.exists(bin_path), f"Binary file not found at path: {bin_path}"
        assert seq_len > 0, f"seq_len must be > 0, got {seq_len}"
        assert batch_size > 0, f"batch_size must be > 0, got {batch_size}"

        self.bin_path = bin_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dtype = dtype

        # Memory map the dataset in read-only mode
        self.data = np.memmap(bin_path, dtype=dtype, mode='r')
        self.total_tokens = len(self.data)
        assert self.total_tokens > seq_len + 1, (
            f"Dataset length ({self.total_tokens}) must be greater than seq_len + 1 ({seq_len + 1})"
        )

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples a random batch of sequences (x, y) with explicit shape assertions.
        x: (batch_size, seq_len) int32
        y: (batch_size, seq_len) int32
        """
        max_idx = self.total_tokens - self.seq_len - 1
        rand_indices = np.random.randint(0, max_idx, size=(self.batch_size,))
        
        x_list = [self.data[i : i + self.seq_len] for i in rand_indices]
        y_list = [self.data[i + 1 : i + 1 + self.seq_len] for i in rand_indices]

        x = np.stack(x_list).astype(np.int32)
        y = np.stack(y_list).astype(np.int32)

        # Boundary Shape Assertions (Hard Rule)
        assert x.shape == (self.batch_size, self.seq_len), (
            f"Expected x shape ({self.batch_size}, {self.seq_len}), got {x.shape}"
        )
        assert y.shape == (self.batch_size, self.seq_len), (
            f"Expected y shape ({self.batch_size}, {self.seq_len}), got {y.shape}"
        )
        assert x.dtype == np.int32, f"Expected x dtype int32, got {x.dtype}"
        assert y.dtype == np.int32, f"Expected y dtype int32, got {y.dtype}"

        return x, y
