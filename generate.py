# generate.py
import torch
from src.toy_qwen3.model import Qwen3Model

# === CONFIG (must match train.py) ===
cfg = {
    "vocab_size": 65,           # Must match training (usually 65)
    "emb_dim": 64,              # Must match
    "n_heads": 4,               # Must match
    "n_kv_groups": 2,
    "n_layers": 2,
    "hidden_dim": 128,
    "context_length": 64,      # Must match (you used 128 now!) note: used 128 for 20 epoch
    "rope_base": 10000.0,
    "qk_norm": False,
    "dtype": torch.bfloat16,
    "head_dim": None,
}

# === DEVICE ===
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# === CHARACTER VOCABULARY (from training) ===
# These are the 65 unique chars in Tiny Shakespeare
chars = '\n !"$&\'(),-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

def decode(tensor):
    return ''.join([idx_to_char[i.item()] for i in tensor])

# === GENERATION FUNCTION ===
def generate(model, prompt, max_new_tokens=100, temperature=0.85):
    model.eval()
    encoded = torch.tensor(
        [char_to_idx.get(c, 0) for c in prompt], 
        dtype=torch.long
    ).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        # Crop to model's context length
        input_ids = encoded[:, -cfg["context_length"]:]
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=cfg["dtype"]):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        encoded = torch.cat([encoded, idx_next], dim=1)
    
    return decode(encoded[0])

# === LOAD MODEL ===
model = Qwen3Model(cfg).to(device).to(cfg["dtype"])
model.load_state_dict(torch.load("toy_qwen3_tiny_shakespeare.pth"))
model.eval()

print("✅ Model loaded and ready for generation!\n")
print("Prompt: 'To be or not to be, that is the'")
print("Generated:", repr(generate(model, "To be ")))