# train.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Import your model components
from src.toy_qwen3.model import Qwen3Model
from src.toy_qwen3.layers import GroupQueryAttention, FeedForward, RMSNormalisation

# ---- 1. Load Tiny Shakespeare ----
print("Loading dataset...")
with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

def encode(s):
    return torch.tensor([char_to_idx[c] for c in s if c in char_to_idx], dtype=torch.long)

def decode(tensor):
    return ''.join([idx_to_char[i.item()] for i in tensor])

data = encode(text)
print(f"Dataset loaded: {len(text):,} chars, vocab_size={vocab_size}")

# ---- 2. Configuration (replaces config.py) ----
cfg = {
    "vocab_size": vocab_size,
    "emb_dim": 64,
    "n_heads": 4,
    "n_kv_groups": 2,        # GQA: 2 KV heads (4 query heads per KV group)
    "n_layers": 2,
    "hidden_dim": 128,
    "context_length": 64,
    "rope_base": 10000.0,
    "qk_norm": False,
    "dtype": torch.bfloat16,
    "head_dim": None,  # automatically set to emb_dim // n_heads = 16
}

# ---- 3. Device (MPS) ----
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Build model
model = Qwen3Model(cfg).to(device).to(cfg["dtype"])
optimizer = AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# ---- 4. Training Setup ----
seq_len = cfg["context_length"]
batch_size = 4
epochs = 8
steps_per_epoch = 100
losses = []

model.train()
print(f"\nStarting training: {epochs} epochs × {steps_per_epoch} steps")

start_time = time.time()

for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")

    for step in progress_bar:
        # Sample random sequences
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        xb = torch.stack([data[i:i+seq_len] for i in ix]).to(device).to(torch.long)
        yb = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device).to(torch.long)

        optimizer.zero_grad()
        with torch.autocast(device_type="cpu", dtype=cfg["dtype"]):
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

# ---- 5. Save Model ----
torch.save(model.state_dict(), "toy_qwen3_tiny_shakespeare.pth")
print("\n🎉 Training complete! Model saved.")

# ---- 6. Save loss plot ----
plt.figure(figsize=(8, 4))
plt.plot(losses, label="Training Loss", marker="o")
plt.title("Toy Qwen3 - Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
print("📉 Loss curve saved as 'loss_curve.png'")

model_save_path = f"toy_qwen3_{epochs}epochs.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to '{model_save_path}'")

plot_save_path = f"loss_curve_{epochs}epochs.png"
plt.savefig(plot_save_path)
print(f"Loss curve saved as '{plot_save_path}'")