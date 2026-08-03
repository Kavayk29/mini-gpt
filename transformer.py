import torch
import torch.nn as nn
import torch.nn.functional as F

from config import n_embd, n_head, block_size, device

head_size = n_embd//n_head

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size,block_size))
        )

    def forward(self, x):
        B, S, E = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        weights = (q @ k.transpose(-2,-1) ) / (self.head_size**0.5 )
        weights = weights.masked_fill(self.tril[:S,:S] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList ( [Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads], dim=-1
        )

        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self):
        super().__init__()

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x  + self.ffwd(self.ln2(x))
        return x