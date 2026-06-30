import torch
import argparse
from model import BigramLanguageModel

# ── args ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='', help='text prompt to start generation from')
parser.add_argument('--max_new_tokens', type=int, default=1000, help='number of tokens to generate')
parser.add_argument('--ckpt', type=str, default='checkpoints/model_3000.pt', help='checkpoint to load')
args = parser.parse_args()

# ── device ───────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── load checkpoint ───────────────────────────────────────────────
checkpoint = torch.load(
    args.ckpt,
    map_location=device,
    weights_only=True
)
stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_size = len(stoi)
block_size = 512
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.1

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# ── load model ────────────────────────────────────────────────────
m = BigramLanguageModel(
    vocab_size,
    block_size,
    n_embd,
    n_head,
    n_layer,
    dropout,
    device
).to(device)
m.load_state_dict(checkpoint['model'])
m.eval()

# ── build starting context from prompt ────────────────────────────
if args.prompt:
    prompt_tokens = encode(args.prompt)
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
else:
    # no prompt — start from zero token
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

# ── generate ──────────────────────────────────────────────────────
out = m.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()
text = decode(out)

# strip everything after the stop token if present
stop_token = "<|endofgame|>"
if stop_token in text:
    text = text.split(stop_token)[0]

print(text)