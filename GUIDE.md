# NeuralNetZero Training Guide

## Quick Start

```bash
# Train the model (~5000 steps on GPU)
python train.py

# Generate text from a trained checkpoint
python inference.py
```

## What You Built

A ~1M parameter GPT-style language model trained on STEM knowledge (physics, chemistry, math, biology). Every technique used is state-of-the-art as of 2025.

## Architecture

```
+--------------------------------------------------+
|  Input tokens (character-level, vocab ~49)        |
|  +--> Token Embedding (49 x 128)                 |
|       Weight-tied with output head                |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|  Transformer Block x5                             |
|  +----------------------------------------------+|
|  | RMSNorm                                      ||
|  | Causal Self-Attention (4 heads, d_head=32)   ||
|  |   - Fused QKV projection (no bias)           ||
|  |   - RoPE position encoding                   ||
|  |   - PyTorch SDPA (Flash-like attention)       ||
|  |   - Output projection (no bias)              ||
|  | + Residual connection                        ||
|  +----------------------------------------------+|
|  | RMSNorm                                      ||
|  | SwiGLU MLP (128 -> 344 -> 128)               ||
|  |   - gate_proj, up_proj, down_proj (no bias)  ||
|  |   - SiLU(gate) * up, then down               ||
|  | + Residual connection                        ||
|  +----------------------------------------------+|
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|  Final RMSNorm                                    |
|  Output Head (128 -> 49) [tied with embeddings]   |
|  +--> Next token probabilities                    |
+--------------------------------------------------+

Total: 995,840 parameters
```

## Parameter Budget

| Component | Shape | Params |
|-----------|-------|--------|
| Token embedding (tied) | 49 x 128 | 6,272 |
| Per-layer attention QKV+O | 128 x 384 + 128 x 128 | 65,536 |
| Per-layer SwiGLU MLP | 128x344 + 128x344 + 344x128 | 132,096 |
| Per-layer RMSNorm x2 | 128 + 128 | 256 |
| Final RMSNorm | 128 | 128 |
| **Per layer total** | | **197,888** |
| **5 layers + embeddings + final norm** | | **995,840** |

## File Map

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | ~35 | `ModelConfig` and `TrainConfig` dataclasses |
| `tokenizer.py` | ~40 | `CharTokenizer` — builds vocab from data, encode/decode |
| `model.py` | ~120 | `GPT` module: RMSNorm, RoPE, CausalSelfAttention, SwiGLUMLP |
| `optim.py` | ~100 | `Muon` optimizer + `build_optimizer` (Muon for 2D, AdamW for rest) |
| `train.py` | ~350 | Main entry point: data loading, training loop, evaluation |
| `inference.py` | ~80 | Load checkpoint, interactive generation |
| `v0_pure_python.py` | ~827 | Original ~800 param pure Python version (preserved) |
| `data/*.jsonl` | ~200 | STEM dataset: physics, chemistry, math, biology |

## SOTA Techniques Explained

### Why each choice matters

**SwiGLU activation** (instead of ReLU/GELU)
- `SiLU(gate(x)) * up(x)` — gated activation with smooth non-linearity
- 3 matrices instead of 2, but better quality per FLOP
- Used by LLaMA, Phi, Gemma, Mistral

**RoPE positions** (instead of learned/sinusoidal)
- Rotary Position Embedding encodes position via rotation in complex space
- Better length generalization than learned embeddings
- No extra parameters — applied directly to Q and K in attention

**Pre-RMSNorm** (instead of LayerNorm / post-norm)
- RMSNorm: normalize by root-mean-square only (no mean subtraction)
- "Pre" = normalize before attention/MLP, not after
- More stable training, especially at scale

**Weight tying**
- `lm_head.weight = tok_emb.weight` — same matrix used for input and output
- Saves vocab_size x d_model parameters (6,272 params here)
- Empirically works as well or better

**No biases anywhere**
- Following LLaMA: all Linear layers have `bias=False`
- Simpler, fewer params, better generalization

**Muon optimizer** (for 2D weight matrices)
- Nesterov momentum + Newton-Schulz orthogonalization
- From modded-nanogpt: ~2x compute efficiency over AdamW for matrices
- Only applied to 2D weights; embeddings/norms use standard AdamW

**Cosine LR schedule with warmup**
- Linear warmup for first 100 steps (avoids early instability)
- Cosine decay to 0 over remaining steps
- Proven schedule used by most modern LLMs

**BF16 mixed precision**
- Forward/backward in bfloat16, optimizer states in float32
- 2x faster, half memory, negligible quality loss
- Your RTX 5070 has native BF16 support

**torch.compile** (disabled on Windows, works on Linux)
- JIT-compiles the model for 10-30% speedup
- Requires Triton (Linux only currently)

**Curriculum learning**
- Dataset sorted by text length (short first)
- Model sees simple patterns before complex ones

## Training Output

When you run `python train.py`, you'll see:

```
Device: cuda | Dtype: torch.bfloat16

Loading data...
  199 examples loaded
  Vocab size: 49 tokens
  Total tokens: 23889

============================================================
MODEL ARCHITECTURE
============================================================
  Layers:     5
  d_model:    128
  Heads:      4 (d_head=32)
  FFN inner:  344 (SwiGLU)
  Context:    512
  Vocab:      49
  Parameters: 995,840
============================================================

TRAINING
============================================================
  Random baseline loss: 3.89

  step     1/5000 | loss 3.8912 [#####.........] | lr 1.0e-4 | 0s
  ...
  step  5000/5000 | loss X.XXXX [.....] | lr 0.0e+0 | XXs
```

- **Random baseline**: ln(49) = 3.89. If your loss starts here, the model is learning.
- **Loss should decrease** steadily. Expect final loss well below 3.0.
- **Eval samples** print every 200 steps so you can watch the model learn.
- A **checkpoint.pt** is saved at the end.

## Interactive Generation

```bash
python inference.py
```

```
> the force
  the force equals mass times acceleration and this...

> :temp 0.5    # lower = more deterministic
> :topk 10     # fewer candidates = more focused
> :quit
```

## Adding More Data

Create a new `.jsonl` file in `data/`:

```json
{"text": "your text here", "domain": "physics", "difficulty": "basic"}
{"text": "longer explanation of a concept goes here", "domain": "physics", "difficulty": "intermediate"}
```

The training script automatically loads all `data/*.jsonl` files.

Length guidelines per domain:
- 8 tiny (2-16 chars): formulas like "f=ma"
- 12 short (16-64 chars): one-line definitions
- 15 medium (64-256 chars): concept explanations
- 15 long (256-512 chars): detailed explanations

## Hyperparameter Tuning

Edit `config.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | 32 | Reduce if OOM, increase for smoother gradients |
| `learning_rate` | 1e-3 | AdamW LR for embeddings/norms |
| `muon_lr` | 0.02 | Muon LR for 2D weights (higher than AdamW) |
| `max_steps` | 5000 | More steps = more training |
| `block_size` | 512 | Context window (reduce if OOM) |
| `warmup_steps` | 100 | Linear LR warmup |
| `grad_clip` | 1.0 | Max gradient norm |

## Scaling Roadmap

| Scale | d_model | Layers | Heads | Params | What it learns |
|-------|---------|--------|-------|--------|----------------|
| **Current** | 128 | 5 | 4 | ~1M | Character patterns, common words |
| Next | 256 | 8 | 8 | ~10M | Grammar, sentence structure |
| Later | 512 | 12 | 8 | ~100M | Knowledge, reasoning |
| Goal | 1024 | 24 | 16 | ~1B | Fluency, deep understanding |

To scale up: increase `d_model`, `n_layer`, `n_head` in `config.py`, add BPE tokenizer, expand dataset.

## Original Pure Python Version

The original ~800 parameter pure Python GPT is preserved as `v0_pure_python.py`. It runs independently with zero dependencies:

```bash
python v0_pure_python.py
```

This demonstrates autograd, backprop, attention, and transformers from scratch — the foundation everything else builds on.
