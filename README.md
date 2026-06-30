# NanoGPT Custom

A from-scratch implementation of a GPT language model, built by following Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) video lecture and then extending it with my own optimizations.

## Overview

This project walks through building a GPT-style transformer from the ground up — starting with a simple bigram model and progressively adding the components that make modern language models work. The model was trained on Steam review data using my personal GPU via CUDA, and AdamW was used to further optimize training.

## What I Learned

Working through this project I got hands-on experience with every major concept in modern language modeling:

- **Tokenization** — character-level encoding, mapping text to integer sequences
- **Bigram Language Model** — the simplest possible next-token predictor, used as a baseline
- **Token & Positional Embeddings** — giving the model both *what* a token is and *where* it appears in the sequence
- **Self-Attention** — the core mechanism that lets tokens communicate with each other
- **Multi-Head Attention** — running multiple attention heads in parallel to capture different relationships
- **Feed-Forward Layers** — the MLP block that processes each token independently after attention
- **Residual Connections** — skip connections that allow gradients to flow cleanly through deep networks
- **Layer Normalization** — stabilizing activations across the network
- **Dropout** — regularization to prevent overfitting
- **The Training Loop** — batching, loss calculation, backpropagation, AdamW optimizer
- **Learning Rate Scheduling** — warmup + cosine decay for stable training
- **Gradient Accumulation** — simulating larger batch sizes on consumer hardware
- **Loss Estimation** — averaging loss over multiple batches for a cleaner validation signal

## Dataset

Training was done on Steam game/review data.

**First attempt — Steam games dataset (~400MB)**
https://www.kaggle.com/datasets/fronkongames/steam-games-dataset 
The initial dataset pulled descriptions, genres, tags, and metadata from a Steam games CSV. While rich in content, this dataset contained **4000+ unique characters** due to HTML artifacts, special symbols, and non-English text mixed throughout. The character-level tokenizer struggled with this vocabulary size and the model's performance suffered as a result.

**Second attempt — Steam reviews dataset (~2.3GB)**
https://www.kaggle.com/datasets/andrewmvd/steam-reviews 
Switched to a Steam reviews dataset that was majority English with a much cleaner character distribution. This worked significantly better — the vocabulary size dropped dramatically and the model produced much more coherent output.

## Training

Trained locally on personal hardware:

- **CPU:** AMD Ryzen 9800X3D
- **GPU:** NVIDIA RTX 4070 Super (12GB VRAM)
- **Framework:** PyTorch with CUDA

CUDA support was added to move both the model and data batches to the GPU, giving a significant speedup over CPU training. AdamW was used as the optimizer to further improve training stability and performance.
Estimated Model Parameter size: 26M parameters, which is actually smaller than a normal GPT-2 by a factor of 6 but, I mostly did this to show that you can create a quick efficent small model that can run on any device no matter the vram.

## File Structure

```
NanoGPT-custom/
├── train.py              # model definition + training loop
├── model.py              # GPT model architecture
├── generate.py           # load a checkpoint and sample text
├── data/
│   └── converted.txt     # cleaned training text
└── checkpoints/          # saved model checkpoints
```

## Requirements

> **Python 3.9 – 3.12 required.** PyTorch does not support Python 3.13+.
> This project was built and tested on **Python 3.9**.

Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy tiktoken tqdm
```

If you have multiple Python versions installed use `py -3.9`:
```bash
py -3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

**Train**
```bash
py -3.9 train.py
```

**Generate**
```bash
# basic generation
py -3.9 generate.py

# with a prompt
py -3.9 generate.py --prompt "Terrible game"

# custom checkpoint and length
py -3.9 generate.py --ckpt checkpoints/model_3000.pt --prompt "The graphics are" --max_new_tokens 500
```

## Acknowledgements

Built by following Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture. Highly recommend it for anyone wanting to understand how transformers actually work under the hood.