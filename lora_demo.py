"""LoRA fine-tuning demo — teaches a small LLM facts about THIS rag_optimized project.

What it shows
-------------
1. A small instruct model answering project questions BEFORE training (it guesses / is generic).
2. We attach LoRA adapters (tiny % of params) and train on ~10 Q&A pairs.
3. The SAME model answering AFTER training (now it knows the project facts).
4. A loss curve saved to lora_out/loss_curve.png.

Run
---
    python3 lora_demo.py                 # uses Qwen2.5-0.5B-Instruct
    MODEL_NAME=... python3 lora_demo.py  # try any small HF instruct model

Hardware
--------
Auto-detects Apple Silicon (MPS) / CUDA / CPU. On Mac it runs in fp32 on MPS.

Tweak it
--------
- TRAIN_DATA below: add your own Q&A pairs (this is your "fine-tuning dataset").
- LORA_R / LORA_ALPHA: adapter size & strength.
- EPOCHS / LR: how hard/long it trains.
Everything is plain PyTorch + PEFT so it's easy to read and change.
"""

import os
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Let unsupported MPS ops fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
OUT_DIR = "lora_out"

# --- LoRA + training hyperparameters (tweak these) ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 40          # tiny dataset -> many passes so it clearly "learns"
LR = 2e-4

# --- The fine-tuning dataset: facts about THIS project ---
# A fresh model has no idea about rag_optimized; after LoRA it answers these.
TRAIN_DATA = [
    {"q": "What chunking strategy does rag_optimized use?",
     "a": "rag_optimized uses structure-aware chunking that splits documents by headings and paragraphs, with a fixed-size overlap fallback for long sections."},
    {"q": "How does rag_optimized rerank results?",
     "a": "It reranks the top retrieved chunks with a cross-encoder model (ms-marco-MiniLM) to reorder them by true relevance before answering."},
    {"q": "What is agentic retrieval in rag_optimized?",
     "a": "Agentic retrieval is a self-reflection loop: the system checks whether the retrieved context is sufficient, issues follow-up searches if not, and only answers once it is grounded."},
    {"q": "How does rag_optimized run without an API key?",
     "a": "It has a demo mode at the /demo/query endpoint that simulates the full pipeline with real guardrails and tracing, so it works with zero configuration."},
    {"q": "What vector database does rag_optimized use?",
     "a": "rag_optimized stores embeddings in ChromaDB and runs parallel similarity searches across multiple query variants."},
    {"q": "What guardrails does rag_optimized apply?",
     "a": "It validates input for prompt injection and PII, and checks output for PII leakage, missing citations, and refusals, redacting PII when needed."},
    {"q": "How is rag_optimized deployed online?",
     "a": "It deploys to Render with a one-click Blueprint using a slim requirements set, and also ships a Dockerfile for Fly, Railway, or local containers."},
    {"q": "What embedding model does rag_optimized use?",
     "a": "It embeds text with the all-MiniLM-L6-v2 sentence-transformers model."},
    {"q": "How does rag_optimized speed up repeated queries?",
     "a": "It caches answers in a user cache and caches retrieved chunks in a chunk cache, both with TTL expiry, to avoid recomputing."},
    {"q": "What observability does rag_optimized provide?",
     "a": "Every pipeline step is wrapped in a timed span; traces and latency metrics are exposed at the /traces and /metrics endpoints."},
]

# Prompts we'll eyeball before vs after training.
PROBE_QUESTIONS = [
    "What chunking strategy does rag_optimized use?",
    "What is agentic retrieval in rag_optimized?",
]


def get_device() -> str:
    """Check hardware first (per ML best practice) and pick the right backend."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_prompt(tokenizer, question: str) -> str:
    """Wrap a question in the model's chat template."""
    messages = [
        {"role": "system", "content": "You are an assistant that knows the rag_optimized project."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tokenizer, question: str, device: str, max_new_tokens: int = 80) -> str:
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def encode_example(tokenizer, question: str, answer: str, device: str):
    """Build input_ids + labels, masking the prompt so loss is only on the answer."""
    prompt = build_prompt(tokenizer, question)
    full = prompt + answer + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    input_ids = torch.tensor([full_ids], device=device)
    labels = torch.tensor([full_ids], device=device).clone()
    labels[0, : len(prompt_ids)] = -100  # ignore the prompt tokens in the loss
    return input_ids, labels


def main():
    device = get_device()
    print(f"\n=== LoRA demo on rag_optimized ===")
    print(f"Device: {device}  |  Model: {MODEL_NAME}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(device)

    # --- BEFORE training ---
    print("----- BEFORE LoRA (base model) -----")
    before = {}
    for q in PROBE_QUESTIONS:
        ans = generate(model, tokenizer, q, device)
        before[q] = ans
        print(f"Q: {q}\nA: {ans}\n")

    # --- Attach LoRA adapters ---
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA attached: training {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.3f}% of the model)\n")

    # --- Train (plain PyTorch loop so it's easy to follow) ---
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    losses = []
    print(f"Training for {EPOCHS} epochs over {len(TRAIN_DATA)} examples...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for ex in TRAIN_DATA:
            input_ids, labels = encode_example(tokenizer, ex["q"], ex["a"], device)
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += out.loss.item()
        avg = epoch_loss / len(TRAIN_DATA)
        losses.append(avg)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:2d}  loss {avg:.4f}")

    # --- AFTER training ---
    model.eval()
    print("\n----- AFTER LoRA (fine-tuned) -----")
    after = {}
    for q in PROBE_QUESTIONS:
        ans = generate(model, tokenizer, q, device)
        after[q] = ans
        print(f"Q: {q}\nA: {ans}\n")

    # --- Save adapter, loss curve, and a results summary ---
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(os.path.join(OUT_DIR, "adapter"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(losses, marker="o", ms=3)
        plt.title("LoRA fine-tuning loss (rag_optimized facts)")
        plt.xlabel("epoch"); plt.ylabel("avg loss"); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=120)
        print(f"Saved loss curve -> {OUT_DIR}/loss_curve.png")
    except Exception as e:
        print(f"(skipped loss plot: {e})")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"model": MODEL_NAME, "device": device, "epochs": EPOCHS,
                   "final_loss": losses[-1], "before": before, "after": after}, f, indent=2)
    print(f"Saved adapter -> {OUT_DIR}/adapter  and results -> {OUT_DIR}/results.json")
    print(f"\nFinal loss: {losses[0]:.3f} -> {losses[-1]:.3f}")


if __name__ == "__main__":
    main()
