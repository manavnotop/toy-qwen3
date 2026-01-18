"""Text generation script for Toy Qwen3 character-level language model."""

import contextlib
import json

import torch

from src.toy_qwen3 import DEFAULT_MODEL_PATH, VOCAB_PATH, Qwen3Model, get_default_config

# Select device - check CUDA first for proper autocast support
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Load character vocabulary
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    chars = json.load(f)

# Create character-index mapping
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}


def decode(tensor: torch.Tensor) -> str:
    """Decode a tensor of token indices to a string."""
    return "".join([idx_to_char[i.item()] for i in tensor])


cfg = get_default_config(len(chars), device)


def generate(
    model: Qwen3Model,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.85,
) -> str:
    """Generate text from a prompt using the model.

    Args:
        model: The Qwen3Model to use for generation.
        prompt: Input prompt string.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (higher = more random).

    Returns:
        Generated text including the prompt.
    """
    model.eval()  # Put model in inference mode

    # Convert input prompt string to tensor of token indices
    encoded = (
        torch.tensor(
            [char_to_idx.get(c, 0) for c in prompt],
            dtype=torch.long,
        )
        .unsqueeze(0)
        .to(device)
    )

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=cfg["dtype"])
        if device == "cuda"
        else contextlib.nullcontext()
    )

    for _ in range(max_new_tokens):
        input_ids = encoded[:, -cfg["context_length"] :]
        with torch.no_grad(), autocast_context:
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        encoded = torch.cat([encoded, idx_next], dim=1)

    return decode(encoded[0])


model = Qwen3Model(cfg).to(device).to(cfg["dtype"])
model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, weights_only=True))
model.eval()

print("Model loaded and ready for generation!\n")

prompt = "To be or not to be, that is the"
print(f"Prompt: {repr(prompt)}")
print("Generated:", repr(generate(model, prompt)))
