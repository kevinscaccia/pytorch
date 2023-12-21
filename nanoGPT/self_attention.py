import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size, masked=False):
        super(SelfAttention, self).__init__()
        #
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.masked = masked
        # masked attention (decoder)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, C) @ (C, H) ->  (B, T, H)
        k = self.query(x) # (B, T, C) @ (C, H) ->  (B, T, H)
        a_weights = q @ k.transpose(-1, -2) # (B, T, H) @ (B, T, H) -> (B, T, T) (attention scores)
        # attention_weights shape is CxC -> representing weights from each C token to all others C
        a_weights = a_weights / (C**(0.5))# scale factor (scaled dot product attention - sqrt(C))
        if self.masked:
            # uses the masking for dont communicate with the future tokens(replace by -inf to apply softmax)
            a_weights = a_weights.masked_fill(self.tril == 0, float('-inf'))
        #
        a_weights = torch.softmax(a_weights, dim=-1) # normalize in the Channels(embedding dim) dimension
        y = a_weights @ self.value(x) # (B, T, T) @ (B, T, C) -> (B, T, C)
        return y
