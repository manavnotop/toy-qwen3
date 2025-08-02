# 🧠 Toy Qwen3 - Character-Level LLM

A **toy implementation of Qwen3 architecture** for learning purposes. This project trains a character-level language model using a simplified version of the Qwen3 transformer architecture. The model learns to generate Shakespeare-like text based on input prompts.

> ⚠️ **Disclaimer**: This project is for **educational purposes only**. It is a minimal and experimental implementation. There might be inaccuracies, oversights, or suboptimal decisions. Please feel free to open issues or share suggestions!

---

## ✨ Features

- Character-level tokenizer
- Configurable transformer architecture (Qwen3-inspired)
- Loss plotted and saved over training epochs
- Generation script with support for temperature sampling
- MPS (Mac) and CUDA GPU support with `autocast` context

---

## 📈 Training Results

Trained for **24 epochs** on Shakespeare dataset (char-level) using MPS device. Final training loss: **~1.509**.

### 🔹 Generated Text

<img src="assets/output.png" alt="Generated Text Example" width="600" />

### 🔹 Loss Curve

<img src="assets/loss_curve_24epochs.png" alt="Training Loss Curve" width="600" />

---

## 🛠️ Usage

### 1. Train the model
```bash
uv run train.py
```

### 2. Generate text
```bash
uv run generate.py
```

---

## 📁 Directory Structure

```
toy-qwen3/
│
├── src/
│   └── toy_qwen3/
│       └── model.py          # Model architecture
│
├── assets/
│   └── loss_curve_24epochs.png
│
├── train.py                  # Training loop
├── generate.py               # Text generation
├── char_vocab.json           # Character-level vocab
├── toy_qwen3_24epochs.pth    # Trained model weights (gitignored)
├── README.md
└── .gitignore
```

---

## 🧾 Resources

This project was inspired and guided by:

- 📘 Blog Post: *"The Big LLM Architecture Comparison"* – [https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison]
- 📓 Jupyter Notebook: *"Qwen3 From Scratch (A Standalone Notebook)"* – [https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb]

---

## 🙏 Acknowledgements

Thanks to the open-source community for resources and ideas. Special thanks to the Qwen team for releasing their architecture and inspiring learning projects like this.

---

## 📌 Notes

- The model weights (`*.pth`) are excluded from the repo.
- Results may vary based on training duration and device.
- The implementation prioritizes readability and learning over efficiency.
