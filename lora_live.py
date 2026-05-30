"""Tiny, REAL LoRA fine-tuning in pure NumPy — runs on free-tier Render (512MB / 0.1 CPU).

Why pure NumPy? PyTorch + any pretrained model needs ~1-2GB RAM and won't fit the
free tier. This is a genuine LoRA setup — a frozen base model plus trainable low-rank
adapters (A @ B) — just scaled down to a small char-level model so it trains live in
the browser request on a hosted free instance.

The model
---------
A char-level next-token predictor. Features are FROZEN (this is the "pretrained base"
we don't touch): char embeddings E, a random tanh projection (Wh, bh). Only the
output weights get a LoRA update:

    logits = (Wo0 + A @ B) @ h + bo      # Wo0 is FROZEN; A, B, bo are trained

So training touches only A, B, bo — a few % of params — exactly the LoRA idea.

Tweak it
--------
- D (hidden size), R (LoRA rank), CTX (context window), STEPS, LR below.
- Add your own featurizer or swap the base init to experiment.
Everything is plain NumPy with hand-written gradients so it's easy to read.
"""

import numpy as np

# --- Hyperparameters (tweak these) ---
D = 48          # hidden size of the frozen base
R = 8           # LoRA rank (how big the low-rank adapter is)
CTX = 4         # how many previous chars the model sees
STEPS = 600     # SGD updates
BATCH = 96      # positions per update
LR = 0.8        # learning rate
MAX_CHARS = 6000  # cap uploaded text so free-tier CPU stays snappy
GEN_LEN = 220   # chars to generate for the before/after samples


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _build_vocab(text):
    chars = sorted(set(text))
    # index 0 is a PAD/BOS token used to left-pad the first characters
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    itos[0] = ""
    return stoi, itos, len(chars) + 1


def _featurize(ids, E, Wh, bh):
    """Precompute the FROZEN features h for every position, plus the targets.

    h depends only on frozen weights, so it's constant during training — we compute
    it once and then training is a fast linear-softmax problem on top of it.
    """
    n = len(ids)
    H = np.zeros((n, D), dtype=np.float32)
    targets = np.array(ids, dtype=np.int64)
    for i in range(n):
        lo = max(0, i - CTX)
        ctx = ids[lo:i]
        emb = E[ctx].mean(axis=0) if ctx else np.zeros(D, dtype=np.float32)
        H[i] = np.tanh(Wh @ emb + bh)
    return H, targets


def _gen_h(ctx_ids, E, Wh, bh):
    emb = E[ctx_ids].mean(axis=0) if ctx_ids else np.zeros(D, dtype=np.float32)
    return np.tanh(Wh @ emb + bh)


def _generate(seed_ids, V, E, Wh, bh, Wo0, A, B, bo, itos, length=GEN_LEN, temp=0.8):
    out_ids = list(seed_ids)
    W = Wo0 + A @ B
    for _ in range(length):
        ctx = out_ids[-CTX:]
        h = _gen_h(ctx, E, Wh, bh)
        logits = W @ h + bo
        p = _softmax(logits / temp)
        nxt = int(np.random.default_rng().choice(V, p=p))
        if nxt == 0:  # PAD/BOS — skip, don't emit
            continue
        out_ids.append(nxt)
    return "".join(itos.get(i, "") for i in out_ids)


def finetune_on_text(text, steps=STEPS, seed_prompt=None):
    """Run a real LoRA fine-tune on `text`. Returns a JSON-friendly dict."""
    text = (text or "").strip()
    if len(text) < 20:
        raise ValueError("Need at least ~20 characters of text to fine-tune on.")
    text = text[:MAX_CHARS]

    rng = np.random.default_rng(0)  # deterministic frozen base
    stoi, itos, V = _build_vocab(text)
    ids = [stoi[c] for c in text]

    # --- FROZEN base weights (the "pretrained" part we never update) ---
    E = (rng.standard_normal((V, D)) * 0.3).astype(np.float32)
    Wh = (rng.standard_normal((D, D)) * (1.0 / np.sqrt(D))).astype(np.float32)
    bh = np.zeros(D, dtype=np.float32)
    Wo0 = (rng.standard_normal((V, D)) * (1.0 / np.sqrt(D))).astype(np.float32)

    # --- Trainable LoRA adapter + output bias (B starts at 0 => delta starts at 0) ---
    A = (rng.standard_normal((V, R)) * 0.1).astype(np.float32)
    B = np.zeros((R, D), dtype=np.float32)
    bo = np.zeros(V, dtype=np.float32)

    trainable = A.size + B.size + bo.size
    total = trainable + E.size + Wh.size + bh.size + Wo0.size

    H, targets = _featurize(ids, E, Wh, bh)
    n = len(ids)

    # Seed for the before/after samples.
    if seed_prompt:
        seed_ids = [stoi[c] for c in seed_prompt if c in stoi][:CTX] or [ids[0]]
    else:
        seed_ids = ids[:CTX]

    # BEFORE: generate with the un-trained adapter (delta == 0 -> pure frozen base).
    before = _generate(seed_ids, V, E, Wh, bh, Wo0, A, B, bo, itos)

    # --- Training loop: minibatch SGD on A, B, bo only ---
    loss_history = []
    for step in range(steps):
        idx = rng.integers(0, n, size=min(BATCH, n))
        h = H[idx]                       # (b, D)
        W = Wo0 + A @ B                  # (V, D)
        logits = h @ W.T + bo            # (b, V)
        p = _softmax(logits)
        tgt = targets[idx]
        loss = -np.log(p[np.arange(len(idx)), tgt] + 1e-9).mean()
        loss_history.append(round(float(loss), 4))

        # gradients (hand-written)
        dlogits = p
        dlogits[np.arange(len(idx)), tgt] -= 1.0
        dlogits /= len(idx)              # (b, V)
        dbo = dlogits.sum(axis=0)        # (V,)
        dW = dlogits.T @ h               # (V, D)  -> grad wrt (Wo0 + A@B)
        dA = dW @ B.T                    # (V, R)
        dB = A.T @ dW                    # (R, D)

        A -= LR * dA
        B -= LR * dB
        bo -= LR * dbo

    # AFTER: same seed, now with the trained adapter.
    after = _generate(seed_ids, V, E, Wh, bh, Wo0, A, B, bo, itos)

    # Log every Nth point so the curve isn't huge.
    keep = max(1, len(loss_history) // 60)
    sparse_loss = loss_history[::keep]

    return {
        "ok": True,
        "model": "char-LoRA (pure NumPy)",
        "device": "render-cpu",
        "chars": len(text),
        "vocab": V,
        "steps": steps,
        "params": {
            "trainable": int(trainable),
            "total": int(total),
            "pct": round(100.0 * trainable / total, 2),
        },
        "loss_history": sparse_loss,
        "loss_start": loss_history[0],
        "loss_end": loss_history[-1],
        "before": before,
        "after": after,
    }


if __name__ == "__main__":
    sample = (
        "Retrieval augmented generation enriches a language model with external "
        "knowledge. The pipeline chunks documents, embeds them, retrieves the most "
        "relevant pieces, reranks them, and then generates a grounded answer. "
    ) * 6
    out = finetune_on_text(sample, steps=300)
    print("trainable %:", out["params"]["pct"])
    print("loss:", out["loss_start"], "->", out["loss_end"])
    print("\nBEFORE:\n", out["before"][:200])
    print("\nAFTER:\n", out["after"][:200])
