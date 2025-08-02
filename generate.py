import torch
import contextlib
import json
from src.toy_qwen3.model import Qwen3Model

#select device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

#load character vocabulary
with open("char_vocab.json", "r") as f:
    chars = json.load(f)

#create character-index mapping
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

#converts tensor of indicies into string
def decode(tensor):
    return ''.join([idx_to_char[i.item()] for i in tensor])

cfg = {
    "vocab_size": len(chars),   #number of unique characters(tokens)
    "emb_dim": 128,             #token embedding dimension
    "n_heads": 4,               #total number of attention heads
    "n_kv_groups": 2,           #group of kv heads
    "n_layers": 4,              #number of transformer layers
    "hidden_dim": 128,          #feedforward hidden dimension
    "context_length": 128,      #max input sequence length
    "rope_base": 10000.0,       #rotary embedding base (for RoPE)
    "qk_norm": False,           #whether to use qk_normalisation
    "dtype": torch.float32 if device == "mps" else torch.bfloat16,         #precision used in model (bfloat16 or float32)
    "head_dim": None,           #will be computed as emb_dim // n_heads
}


def generate(model, prompt, max_new_tokens=100, temperature=0.85):
    model.eval() #put model in inference mode

    #convert input prompt string to tensor of token indices
    encoded = torch.tensor(
        [char_to_idx.get(c, 0) for c in prompt], 
        dtype=torch.long
    ).unsqueeze(0).to(device)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=cfg["dtype"])
        if torch.cuda.is_available() else contextlib.nullcontext()
    )

    for _ in range(max_new_tokens):
        input_ids = encoded[:, -cfg["context_length"]:]
        with torch.no_grad(), autocast_context:
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        encoded = torch.cat([encoded, idx_next], dim=1)

    return decode(encoded[0])

model = Qwen3Model(cfg).to(device).to(cfg["dtype"])
#PLEASE MAKE SURE YOU LOAD THE RIGHT MODEL ACCORIDNG TO THE EPOCHS YOU USED AND THE NAME WITH WHICH MODEL IS SAVED
model.load_state_dict(torch.load("toy_qwen3_24epochs.pth"))
model.eval()

print("✅ Model loaded and ready for generation!\n")

prompt = "To be or not to be, that is the"
print(f"Prompt: {repr(prompt)}")
print("Generated:", repr(generate(model, prompt)))