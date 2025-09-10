import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        # Handle odd d_model
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # Assign sin and cos terms
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:  # If d_model is odd, exclude the last unmatched position
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # Return position encodings for the input length

    
if __name__ == '__main__':
    # pe = PositionalEmbedding(20)
    # y = pe.forward(torch.zeros(1, 100))
    # print(y.size())
    import torch

    # Initialize parameters
    d_model = 64   # Dimensionality of the embedding
    seq_len = 50   # Sequence length for testing
    batch_size = 32  # Number of sequences in the batch

    # Create an instance of the PositionalEmbedding class
    pos_emb = PositionalEmbedding(d_model)

    # Generate a test input tensor of shape (batch_size, seq_len, d_model)
    test_input = torch.zeros((batch_size, seq_len, d_model))

    # Pass the test input through the positional embedding module
    output = pos_emb(test_input)

    # Verify the output shape
    print("Output shape:", output.shape)  # Expected: (1, seq_len, d_model)

    # Inspect part of the output
    print("Output positional embedding (first sequence, first 5 positions):")
    print(output[0, :5, :])
