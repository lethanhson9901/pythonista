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
        
        # Linear transformations for queries, keys, and values project the input x into different subspaces, 
        # creating different sets of queries, keys, and values for each attention head.
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

        #  where batch_size is the number of sequences in a batch, 
        # seq_len is the length of each sequence,
        #  _ denotes the feature dimension, which can vary depending on the problem.
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations for queries, keys, and values. Each tensor has dimensions of (batch_size, seq_len, num_heads, head_dim).
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


class ResidualConnection(nn.Module):
    """
    A ResidualConnection module combines a sub-layer's output with the original input using layer normalization.

    Args:
        sublayer (nn.Module): The sub-layer to apply.
        d_model (int): The dimension of the model.
        dropout (float): Dropout probability (default is 0.1).
    """
    def __init__(self, sublayer, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.sublayer = sublayer
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the ResidualConnection module.

        Args:
            x (Tensor): The input tensor.
            *args: Additional arguments for the sub-layer.
            **kwargs: Additional keyword arguments for the sub-layer.

        Returns:
            Tensor: The output tensor with the residual connection applied.
        """
        sublayer_output = self.sublayer(x, *args, **kwargs)
        normalized_output = self.layer_norm(x + self.dropout(sublayer_output))
        return normalized_output

class EncoderBlock(nn.Module):
    """
    An EncoderBlock consists of a Multi-Head Self-Attention layer and a Feed-Forward Block.

    Args:
        embed_dim (int): Dimension of the input data.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the hidden layer in the feed-forward block.
        dropout_prob (float): Dropout probability (e.g., 0.1 for 10% dropout).
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_prob=0.1):
        super(EncoderBlock, self).__init()
        self.self_attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout_prob)
        self.feed_forward = FeedForwardBlock(embed_dim, hidden_dim, dropout_prob)
        self.residual_connection_attention = ResidualConnection(self.self_attention, embed_dim, dropout_prob)
        self.residual_connection_feed_forward = ResidualConnection(self.feed_forward, embed_dim, dropout_prob)

    def forward(self, x, mask=None):
        """
        Forward pass of the EncoderBlock module.

        Args:
            x (Tensor): Input tensor.
            mask (Tensor, optional): Mask to apply to the attention scores (optional).

        Returns:
            Tensor: Output of the EncoderBlock.
        """
        # Apply multi-head self-attention with residual connection and layer normalization
        attention_output = self.residual_connection_attention(x, mask=mask)
        
        # Apply feed-forward block with residual connection and layer normalization
        encoder_output = self.residual_connection_feed_forward(attention_output)
        
        return encoder_output

class Encoder(nn.Module):
    """
    The Encoder is composed of a stack of EncoderBlocks.

    Args:
        num_blocks (int): Number of EncoderBlocks in the stack.
        embed_dim (int): Dimension of the input data.
        num_heads (int): Number of attention heads in each EncoderBlock.
        hidden_dim (int): Dimension of the hidden layer in the feed-forward block.
        vocab_size (int): Vocabulary size for the input.
        max_seq_len (int): Maximum sequence length (default is 512).
        dropout_prob (float): Dropout probability (e.g., 0.1 for 10% dropout).
    """
    def __init__(self, num_blocks, embed_dim, num_heads, hidden_dim, vocab_size, max_seq_len=512, dropout_prob=0.1):
        super(Encoder, self).__init()
        self.embedding = InputEmbeddings(embed_dim, vocab_size)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads, hidden_dim, dropout_prob) for _ in range(num_blocks)])

    def forward(self, x, mask=None):
        """
        Forward pass of the Encoder module.

        Args:
            x (Tensor): Input tensor.
            mask (Tensor, optional): Mask to apply to the attention scores (optional).

        Returns:
            Tensor: Output of the Encoder.
        """
        # Apply input embeddings and positional encodings
        embedded_input = self.positional_encoding(self.embedding(x))
        
        # Apply the stack of EncoderBlocks
        for block in self.blocks:
            embedded_input = block(embedded_input, mask=mask)
        
        return embedded_input

class DecoderBlock(nn.Module):
    """
    A DecoderBlock in the Transformer architecture consists of masked multi-head self-attention,
    encoder-decoder multi-head attention, and a feed-forward block.

    Args:
        embed_dim (int): Dimension of the input data.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the hidden layer in the feed-forward block.
        dropout_prob (float): Dropout probability (e.g., 0.1 for 10% dropout).
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_prob=0.1):
        super(DecoderBlock, self).__init()
        self.self_attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout_prob)
        self.encoder_decoder_attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout_prob)
        self.feed_forward = FeedForwardBlock(embed_dim, hidden_dim, dropout_prob)
        self.residual_connection_self_attention = ResidualConnection(self.self_attention, embed_dim, dropout_prob)
        self.residual_connection_encoder_decoder_attention = ResidualConnection(self.encoder_decoder_attention, embed_dim, dropout_prob)
        self.residual_connection_feed_forward = ResidualConnection(self.feed_forward, embed_dim, dropout_prob)
    
    def forward(self, x, encoder_output, self_mask=None, encoder_decoder_mask=None):
        """
        Forward pass of the DecoderBlock module.

        Args:
            x (Tensor): Input tensor.
            encoder_output (Tensor): Output from the encoder.
            self_mask (Tensor, optional): Mask to apply to the self-attention scores (optional).
            encoder_decoder_mask (Tensor, optional): Mask to apply to the encoder-decoder attention scores (optional).

        Returns:
            Tensor: Output of the DecoderBlock.
        """
        # Apply masked self-attention with residual connection and layer normalization
        self_attention_output = self.residual_connection_self_attention(x, mask=self_mask)
        
        # Apply encoder-decoder attention with residual connection and layer normalization
        encoder_decoder_attention_output = self.residual_connection_encoder_decoder_attention(self_attention_output, encoder_output, mask=encoder_decoder_mask)
        
        # Apply feed-forward block with residual connection and layer normalization
        decoder_output = self.residual_connection_feed_forward(encoder_decoder_attention_output)
        
        return decoder_output

class Decoder(nn.Module):
    """
    The Decoder in the Transformer architecture is composed of a stack of DecoderBlocks.

    Args:
        num_blocks (int): Number of DecoderBlocks in the stack.
        embed_dim (int): Dimension of the input data.
        num_heads (int): Number of attention heads in each DecoderBlock.
        hidden_dim (int): Dimension of the hidden layer in the feed-forward block.
        vocab_size (int): Vocabulary size for the output.
        max_seq_len (int): Maximum sequence length (default is 512).
        dropout_prob (float): Dropout probability (e.g., 0.1 for 10% dropout).
    """
    def __init__(self, num_blocks, embed_dim, num_heads, hidden_dim, vocab_size, max_seq_len=512, dropout_prob=0.1):
        super(Decoder, self).__init()
        self.embedding = InputEmbeddings(embed_dim, vocab_size)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.blocks = nn.ModuleList([DecoderBlock(embed_dim, num_heads, hidden_dim, dropout_prob) for _ in range(num_blocks)])
    
    def forward(self, x, encoder_output, self_mask=None, encoder_decoder_mask=None):
        """
        Forward pass of the Decoder module.

        Args:
            x (Tensor): Input tensor.
            encoder_output (Tensor): Output from the encoder.
            self_mask (Tensor, optional): Mask to apply to the self-attention scores (optional).
            encoder_decoder_mask (Tensor, optional): Mask to apply to the encoder-decoder attention scores (optional).

        Returns:
            Tensor: Output of the Decoder.
        """
        # Apply input embeddings and positional encodings
        embedded_input = self.positional_encoding(self.embedding(x))
        
        # Apply the stack of DecoderBlocks
        for block in self.blocks:
            embedded_input = block(embedded_input, encoder_output, self_mask=self_mask, encoder_decoder_mask=encoder_decoder_mask)
        
        return embedded_input

class ProjectionLayer(nn.Module):
    """
    A ProjectionLayer in the Transformer architecture projects decoder output to the output vocabulary space.

    Args:
        embed_dim (int): Dimension of the input data.
        vocab_size (int): Vocabulary size for the output.
    """
    def __init__(self, embed_dim, vocab_size):
        super(ProjectionLayer, self).__init()
        self.embedding = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        """
        Forward pass of the ProjectionLayer module.

        Args:
            x (Tensor): Input tensor representing decoder output.

        Returns:
            Tensor: Output tensor with vocabulary distribution.
        """
        output_distribution = self.embedding(x)
        return output_distribution
