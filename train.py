import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import contextlib

from src.toy_qwen3.model import Qwen3Model

#load the dataset
print("Loading dataset...")
with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()

#build character level vocabulary
chars = sorted(list(set(text)))
print("chars =", repr(chars))
vocab_size = len(chars)

#mapping characters and token indicies
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

def encode(s):
    return torch.tensor([char_to_idx[c] for c in s if c in char_to_idx], dtype=torch.long)

def decode(tensor):
    return ''.join([idx_to_char[i.item()] for i in tensor])

#encode dataset into tokens ids
data = encode(text)
print(f"Dataset loaded: {len(text):,} chars, vocab_size={vocab_size}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

#uses float32 for MPS(better on MPS), otherwise bfloat16
cfg_dtype = torch.float32 if device == "mps" else torch.bfloat16

cfg = {
    "vocab_size": vocab_size,   #number of unique characters(tokens)
    "emb_dim": 128,             #token embedding dimension
    "n_heads": 4,               #total number of attention heads
    "n_kv_groups": 2,           #group of kv heads
    "n_layers": 4,              #number of transformer layers
    "hidden_dim": 128,          #feedforward hidden dimension
    "context_length": 128,      #max input sequence length
    "rope_base": 10000.0,       #rotary embedding base (for RoPE)
    "qk_norm": False,           #whether to use qk_normalisation
    "dtype": cfg_dtype,         #precision used in model (bfloat16 or float32)
    "head_dim": None,           #will be computed as emb_dim // n_heads
}

model = Qwen3Model(cfg).to(device)
if device != "mps": 
    model = model.to(cfg["dtype"])
optimizer = AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

seq_len = cfg["context_length"]
batch_size = 4
epochs = 24
steps_per_epoch = 250
losses = []

model.train()
print(f"\nStarting training: {epochs} epochs × {steps_per_epoch} steps")

start_time = time.time()

#use automatic mixed precision if using CUDA
use_autocast = torch.cuda.is_available()
autocast_context = (
    torch.autocast(device_type="cuda", dtype=cfg["dtype"])
    if use_autocast else contextlib.nullcontext()
)

#training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")

    for step in progress_bar:
        # Sample random sequences
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        xb = torch.stack([data[i:i+seq_len] for i in ix]).to(device).long()
        yb = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device).long()

        optimizer.zero_grad()

        #forward pass
        with autocast_context:
            logits = model(xb)
            loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

    avg_loss = epoch_loss / steps_per_epoch
    losses.append(avg_loss)
    elapsed = time.time() - start_time
    print(f"✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.3f} | Time: {elapsed:.0f}s")

    #sample generation
    model.eval()
    start = torch.randint(len(data) - seq_len, (1,)).item()
    input_seq = data[start : start + seq_len].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_seq)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    generated = decode(torch.cat([input_seq[0], next_token[0]]))
    print("📝 Sample generation:", repr(generated))
    model.train()

print("\n🎉 Training complete! Model saved.")

#plot the loss curve
plt.figure(figsize=(8, 4))
plt.plot(losses, label="Training Loss", marker="o")
plt.title("Toy Qwen3 - Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
print("📉 Loss curve saved as 'loss_curve.png'")

model_save_path = f"toy_qwen3_{epochs}epochs.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to '{model_save_path}'")

plot_save_path = f"assets/loss_curve_{epochs}epochs.png"
plt.savefig(plot_save_path)
print(f"Loss curve saved as '{plot_save_path}'")

import json
with open("char_vocab.json", "w") as f:
    json.dump(chars, f)
