"""Training script for Toy Qwen3 character-level language model."""

import contextlib
import json
import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from rich import print as rprint
from rich.console import Console
from rich.theme import Theme
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.toy_qwen3 import (
    DATA_PATH,
    VOCAB_PATH,
    Qwen3Model,
    create_qwen3_config,
)

# Rich theme for beautiful logging
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "loss": "magenta",
        "lr": "yellow",
    }
)
console = Console(theme=custom_theme)

# Load the dataset
rprint("[bold cyan]Loading dataset...[/]")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Build character-level vocabulary
chars = sorted(list(set(text)))
console.print(f"[info]Vocabulary size:[/] {len(chars)} characters")
vocab_size = len(chars)

# Mapping characters and token indices
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}


def encode(s: str) -> torch.Tensor:
    """Encode a string to a tensor of token indices."""
    return torch.tensor([char_to_idx[c] for c in s if c in char_to_idx], dtype=torch.long)


def decode(tensor: torch.Tensor) -> str:
    """Decode a tensor of token indices to a string."""
    return "".join([idx_to_char[i.item()] for i in tensor])


# Encode dataset into token ids
data = encode(text)
console.print(f"[info]Dataset loaded:[/] {len(text):,} chars, [info]vocab_size:[/] {vocab_size}")

# Select device - check CUDA first for proper autocast support
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
console.print(f"[info]Using device:[/] {device}")

# Create config (matches LLaMA32 configuration)
cfg = create_qwen3_config(vocab_size=vocab_size, device=device)

model = Qwen3Model(cfg).to(device)
if device != "mps":
    model = model.to(cfg.dtype)

optimizer = AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# Training hyperparameters (matching LLaMA32)
seq_len = cfg.context_length
batch_size = 8
epochs = 24
steps_per_epoch = 250
total_steps = epochs * steps_per_epoch
losses = []

# Learning rate scheduler with warmup and cosine decay
warmup_steps = 100


def lr_lambda(step: int) -> float:
    """Learning rate schedule: linear warmup + cosine decay."""
    if step < warmup_steps:
        return step / warmup_steps
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


scheduler = LambdaLR(optimizer, lr_lambda)

model.train()
rprint(f"\n[bold cyan]Starting training:[/] {epochs} epochs × {steps_per_epoch} steps")

start_time = time.time()

# Use automatic mixed precision for CUDA
use_autocast = device == "cuda"
autocast_context = (
    torch.autocast(device_type="cuda", dtype=cfg.dtype)
    if use_autocast
    else contextlib.nullcontext()
)

# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}")

    for step in progress_bar:
        # Sample random sequences
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        xb = torch.stack([data[i : i + seq_len] for i in ix]).to(device).long()
        yb = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device).long()

        optimizer.zero_grad()

        # Forward pass
        with autocast_context:
            logits = model(xb)
            loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update progress bar with current loss
        current_lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{current_lr:.2e}"})

        epoch_loss += loss.item()

    avg_loss = epoch_loss / steps_per_epoch
    losses.append(avg_loss)
    elapsed = time.time() - start_time
    current_lr = optimizer.param_groups[0]["lr"]

    # Beautiful epoch summary
    console.print(
        f"[cyan]Epoch {epoch + 1:2d}[/] | "
        f"[magenta]Loss: {avg_loss:.3f}[/] | "
        f"[lr]LR: {current_lr:.2e}[/] | "
        f"[success]Time: {elapsed:.0f}s[/]"
    )

    # Sample generation
    model.eval()
    start = torch.randint(len(data) - seq_len, (1,)).item()
    input_seq = data[start : start + seq_len].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_seq)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    generated = decode(torch.cat([input_seq[0], next_token[0]]))
    console.print(f"[info]Sample:[/] {repr(generated[:80])}...")
    model.train()

rprint("\n[bold green]Training complete![/] Model saved.")

# Plot the loss curve
plt.figure(figsize=(8, 4))
plt.plot(losses, label="Training Loss", marker="o")
plt.title("Toy Qwen3 - Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
console.print("[info]Loss curve saved as 'loss_curve.png'[/]")

model_save_path = f"toy_qwen3_{epochs}epochs.pth"
torch.save(model.state_dict(), model_save_path)
console.print(f"[success]Model saved to '{model_save_path}'[/]")

plot_save_path = f"assets/loss_curve_{epochs}epochs.png"
plt.savefig(plot_save_path)
console.print(f"[info]Loss curve saved as '{plot_save_path}'[/]")

with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(chars, f)
