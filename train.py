import torch
import torch.nn as nn
from torch.nn import functional as F

# ── device ──────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')

# ── hyperparameters ──────────────────────────────────────────────
batch_size = 32   # how many independent sequences will we process in parallel?
block_size = 8    # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
n_embd = 32

# ── data ────────────────────────────────────────────────────────
with open('data/converted.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#print('length:', len(text))
#print(text[:1000])
#unique characters that occur in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(''.join(chars))
#print(vocab_size) this came out to 4030 unique characters
#map character to ints for tokenization
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
#print(encode("hii there"))
#print(decode(encode("hii there")))
#get the data tensor
data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000])
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
train_data[:block_size+1]
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    #print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)

# ── data loader ──────────────────────────────────────────────────
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)  # move to GPU
    return x, y

xb, yb = get_batch('train')
#print('inputs:')
#print(xb.shape)
#print(xb)
#print('targets:')
#print(yb.shape)
#print(yb)
#print("----")
for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        #print(f"when input is {context.tolist()} the target: {target}")

torch.manual_seed(1337)

# ── loss estimation ───────────────────────────────────────────────
# averages loss over multiple batches for a cleaner reading than single batch loss
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# ── model ────────────────────────────────────────────────────────
class BigramLanguageModel(nn.Module):
    def __init__(self): #we got a massive vocab size 4030
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # positional encoding — each position (0 to block_size-1) gets its own embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # linear head that projects from n_embd back up to vocab_size for predictions
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,n_embd) token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,n_embd) positional embeddings
        x = tok_emb + pos_emb  # (B,T,n_embd) add together
        logits = self.lm_head(x)  # (B,T,vocab_size) project to vocab

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens so positional embeddings don't go out of range
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# fixed: no args needed since vocab_size is global
m = BigramLanguageModel().to(device)  # move model to GPU
logits, loss = m(xb, yb)
#print(logits.shape)
#print(loss)
#print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# ── training loop ────────────────────────────────────────────────
#pytorch optimization
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # evaluate loss on train and val every eval_interval steps
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f'final loss: {loss.item():.4f}')

# ── generate ─────────────────────────────────────────────────────
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx=context, max_new_tokens=400)[0].tolist()))