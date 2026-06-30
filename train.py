import torch
import torch._dynamo
torch._dynamo.disable()
import os
from model import BigramLanguageModel


# ── device ──────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')

# ── hyperparameters ──────────────────────────────────────────────
batch_size = 16
block_size = 512
max_iters = 3500
eval_interval = 300
eval_iters = 200

n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.1

learning_rate = 3e-4

# ── AMP (mixed precision) ────────────────────────────────────────
use_amp = device == 'cuda'
scaler = torch.amp.GradScaler('cuda')

# ── data ────────────────────────────────────────────────────────
with open('data/converted.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ── batch loader ────────────────────────────────────────────────
def get_batch(split):
    data_split = train_data if split == 'train' else val_data

    ix = torch.randint(len(data_split) - block_size, (batch_size,))

    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

# ── model ────────────────────────────────────────────────────────
m = BigramLanguageModel(
    vocab_size,
    block_size,
    n_embd,
    n_head,
    n_layer,
    dropout,
    device
).to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# ── loss estimate ───────────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)

            with torch.amp.autocast('cuda', enabled=use_amp):
                _, loss = m(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    m.train()
    return out

# ── training loop ────────────────────────────────────────────────
os.makedirs('checkpoints', exist_ok=True)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch('train')

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast('cuda', enabled=use_amp):
        logits, loss = m(xb, yb)

    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    if iter % 1000 == 0:
        torch.save(
            {
                'model': m.state_dict(),
                'stoi': stoi,
                'itos': itos
            },
            f'checkpoints/model_{iter}.pt'
        )

print(f'final loss: {loss.item():.4f}')