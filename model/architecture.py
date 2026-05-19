import torch
import torch.nn as nn
from torch.nn import functional as F


# --- Hyperparameters ---
# Brain size
VOCAB_SIZE = 50257    # GPT-2 Tokenizer
N_EMBD = 64           # embedding dimensions
BLOCK_SIZE = 128      # maximum context window (how many tokens at once)
N_HEAD = 4            # independent attention heads running in parallel
N_LAYER = 3           # Transformer Blocks (how deep the brain is)


class Head(nn.Module):
    # One head of self-attention
    def __init__(self, head_size):
        super().__init__()
        # The linear layers that generate the Query, Key, and Value
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        
        # 'tril' creates a lower-triangular matrix of 1s and 0s
        # register it as a buffer so PyTorch knows it ain't a parameter to be trained
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)   # (Batch, Time, head_size)
        q = self.query(x) # (Batch, Time, head_size)
        
        # Compute attention scores ("affinities")
        # Multiply Q and K transpose and scale down by square root of head_size
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        
        # Apply the mask. This forces the future tokens to be -infinity (which softmax turns to 0)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax turns scores into percentages that add up to 100%
        wei = F.softmax(wei, dim=-1) 
        
        v = self.value(x) # (Batch, Time, head_size)
        
        # Multiply percentages by actual Values to get final output
        out = wei @ v 
        return out


class MultiHeadAttention(nn.Module):
    # Multiple heads of self-attention running in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        # create a list of 'num_heads' independent attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # A final linear layer to mix the outputs of all heads together
        self.proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        # Run all heads in parallel and concatenate their results
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    # simple linear layer followed by a non linearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # hidden layer is typically 4x larger than the embedding (in transformers)
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # LayerNorm keeps mathematical values from getting too big or too small
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # "x +"" are the residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        # transformer blocks
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD) # Final layer norm
        
        # final layer that transforms 64-dimensional meaning back into vocabulary predictions
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) 
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos) 
        
        x = tok_emb + pos_emb     # Add meaning and time
        x = self.blocks(x)        # Run through deep Transformer blocks
        x = self.ln_f(x)          # Normalize
        logits = self.lm_head(x)  # Shape: (Batch, Time, VOCAB_SIZE)
        
        # Calculate loss
        loss = None
        if targets is not None:
            # Reshape tensors to match PyTorch's cross_entropy requirements
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss









