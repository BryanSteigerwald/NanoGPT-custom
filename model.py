import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()

        # key, query, value projections
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # causal mask (lower triangular)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # attention scores
        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        # causal masking (no peeking into future)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted aggregation of values
        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()

        head_size = n_embd // n_head

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(n_head)
        ])

        # projection back to embedding dimension
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # projection + dropout
        out = self.proj(out)
        out = self.dropout(out)

        return out


class Block(nn.Module):
    """ transformer block (pre-norm) """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()

        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)

        # pre-layernorm (stability improvement)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_layer,
        dropout,
        device
    ):
        super().__init__()

        # store config
        self.block_size = block_size
        self.device = device

        # token embeddings (vocab -> embedding space)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # positional embeddings (position -> embedding space)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        # final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # language modeling head (embedding -> vocab)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying (important improvement)
        self.lm_head.weight = self.token_embedding_table.weight

        # weight initialization (Karpathy-style)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ initialize weights like GPT-2 """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # token embeddings
        tok_emb = self.token_embedding_table(idx)

        # positional embeddings
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )

        # combine embeddings
        x = tok_emb + pos_emb

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # final norm
        x = self.ln_f(x)

        # logits over vocabulary
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """ generate text autoregressively """

        for _ in range(max_new_tokens):

            # crop context to block size
            idx_cond = idx[:, -self.block_size:]

            # forward pass
            logits, _ = self(idx_cond)

            # focus on last time step
            logits = logits[:, -1, :]

            # probabilities
            probs = F.softmax(logits, dim=-1)

            # sample next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # append token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx