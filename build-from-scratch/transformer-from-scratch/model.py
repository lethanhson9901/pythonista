import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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
    """
    PositionalEncoding adds positional information to input embeddings
    for Transformer models.

    Args:
        d_model (int): Dimension of the model's embeddings.
        max_seq_len (int): Maximum sequence length for which positional
            encodings will be created (default is 512).
    """
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)
        
        # Create a matrix to hold positional information
        pe = torch.zeros(max_seq_len, d_model)
        
        # Create a list of numbers from 0 to max_seq_len
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        
        # Create a set of values that follow a pattern using sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        
        # Calculate positional encodings based on the pattern
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a dimension to the positional encoding and make it a model buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.

        Args:
            x (Tensor): Input sequence with embeddings.

        Returns:
            Tensor: Input sequence with positional encodings added.
        """
        # Add the positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    """
    LayerNormalization applies layer-wise normalization to input activations.

    Args:
        features (int): Number of features (i.e., the dimensionality of the input).
        eps (float): A small constant added for numerical stability (default is 1e-6).
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.features = features
        self.eps = eps
        
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        """
        Forward pass of the LayerNormalization module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, features).

        Returns:
            Tensor: Normalized output tensor.
        """
        # Calculate mean and standard deviation along the feature dimension
        mean = x.mean(dim=-1, keepdim=True) #calculate the mean along the feature dimension
        std = x.std(dim=-1, keepdim=True)
        
        # Apply layer normalization
        normalized_x = self.gamma * (x - mean) / (std + self.eps) + self.beta
        
        return normalized_x

class FeedForwardBlock(nn.Module):
    """
    A FeedForwardBlock applies a feed-forward transformation to input data.

    Args:
        input_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden layer.
        dropout_prob (float): Dropout probability (e.g., 0.1 for 10% dropout).
    """
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        super(FeedForwardBlock, self).__init()
        
        # First linear layer with ReLU activation
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Second linear layer
        self.linear2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        """
        Forward pass of the FeedForwardBlock module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Transformed output tensor.
        """
        # Apply the first linear layer and ReLU activation
        x = self.linear1(x)
        x = self.relu(x)
        
        # Apply dropout: prevent overfitting and improve the generalization of the model
        x = self.dropout(x)
        
        # Apply the second linear layer
        x = self.linear2(x)
        
        return x

class MultiHeadAttentionBlock(nn.Module):
    """
    A MultiHeadAttentionBlock applies multi-head self-attention to input data.

    Args:
        embed_dim (int): Dimension of the input data.
        num_heads (int): Number of attention heads.
        dropout_prob (float): Dropout probability (e.g., 0.1 for 10% dropout).
    """
    def __init__(self, embed_dim, num_heads, dropout_prob=0.1):
        super(MultiHeadAttentionBlock, self).__init()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        
        # Linear transformations for queries, keys, and values
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Linear transformation for the output
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x, mask=None):
        """
        Forward pass of the MultiHeadAttentionBlock module.

        Args:
            x (Tensor): Input tensor.
            mask (Tensor, optional): Mask to apply to the attention scores (optional).

        Returns:
            Tensor: Multi-head self-attention output.
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations for queries, keys, and values
        queries = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to prepare for matrix multiplication
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 3, 1)
        values = values.permute(0, 2, 1, 3)
        
        # Calculate scaled dot-product attention
        attention_scores = torch.matmul(queries, keys) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values
        attention_output = torch.matmul(attention_weights, values)
        
        # Transpose and reshape
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Linear transformation for the output
        multihead_output = self.W_o(attention_output)
        
        return multihead_output