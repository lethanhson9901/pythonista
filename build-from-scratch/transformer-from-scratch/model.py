import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

import math

import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes InputEmbeddings module.

        Args:
        d_model (int): Embedding dimension.
        vocab_size (int): Vocabulary size.
        """
        super().__init__()
        self.d_model = d_model  # Embedding dimension
        self.vocab_size = vocab_size  # Vocabulary size
        self.embedding = nn.Embedding(vocab_size, d_model)
        """
        Creates an embedding layer for token-to-vector conversion.
        """

    def forward(self, x):
        """
        Defines the forward pass for InputEmbeddings.

        Args:
        x (Tensor): Input tensor of shape (batch_size, sequence_length) containing token IDs.

        Returns:
        Tensor: Output tensor of shape (batch_size, sequence_length, d_model) representing token embeddings.
        """
        # Pass token IDs through the embedding layer.
        # Resulting shape: (batch_size, sequence_length, d_model).
        embeddings = self.embedding(x)

        # Scale embeddings according to Transformer-like best practices.
        # Scaling factor: sqrt(d_model)
        scaled_embeddings = embeddings * math.sqrt(self.d_model)

        # Return scaled embeddings.
        return scaled_embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Initialize the PositionalEncoding module.

        Args:
            d_model (int): Dimensionality of model embeddings.
            seq_len (int): Maximum sequence length for positional encoding.
            dropout (float): Dropout rate to apply to the output.

        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) for positional encoding
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len) for position information
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)

        # Create a vector of shape (d_model) for division terms
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)

        # Apply sine to even indices of the positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model)))

        # Apply cosine to odd indices of the positional encoding
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model)))

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass for the PositionalEncoding module.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor: Output tensor with positional encoding applied.

        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)

