# NeuralNetZero

Best intelligence-per-parameter LLM trained on first-principles STEM knowledge.
Hardware: RTX 5070 Super (16GB VRAM, Blackwell).

## The Algorithm (decision sequence for every change)

1. Question the requirement — does this create learning value NOW?
2. Delete — remove unnecessary complexity first
3. Simplify — fewer files, fewer abstractions
4. Accelerate — optimize only validated bottlenecks
5. Automate — only proven stable workflows

## Principles

- Ship working code, not clever code
- Every parameter must be explainable
- Scale gradually: 1M -> 10M -> 100M+ params
- No dependencies until we need them

## Architecture (~1M params)

```
n_layer=5, d_model=128, n_head=4, d_head=32
FFN: SwiGLU, inner_dim=344 (2.67x ratio)
Positions: RoPE
Norm: Pre-RMSNorm
Activation: SwiGLU
Weight tying: lm_head = tok_emb
Biases: None
Tokenizer: Character-level (vocab ~70)
Context: 512 tokens
Total: ~997K parameters
```

## SOTA Techniques

| Technique | Source |
|-----------|--------|
| SwiGLU activation | LLaMA, Phi |
| RoPE positions | LLaMA |
| Pre-RMSNorm | Universal standard |
| Weight tying | Saves V*E params |
| No biases | LLaMA |
| Muon optimizer (2D weights) | modded-nanogpt, Moonlight |
| AdamW (embeddings, norms) | Standard |
| Cosine LR + warmup | Standard |
| Gradient clipping (1.0) | Universal |
| BF16 mixed precision | 2x speed |
| torch.compile | 10-30% free speedup |
| PyTorch SDPA | Efficient attention |
| Xavier init | Stable training |
| Curriculum learning | Short examples first |

## Project Structure

- `v0_pure_python.py` — Original ~800 param pure Python GPT (preserved)
- `config.py` — Model + training hyperparameters
- `tokenizer.py` — Character tokenizer (+ future BPE)
- `model.py` — GPT nn.Module (RoPE, SwiGLU, RMSNorm)
- `optim.py` — Muon optimizer implementation
- `train.py` — Main training script (entry point)
- `inference.py` — Interactive text generation
- `data/*.jsonl` — STEM dataset (~200 examples across 4 domains)

## Usage

```bash
# Train
python train.py

# Generate text interactively
python inference.py

# Run original pure Python version
python v0_pure_python.py
```

## Dataset Format

Each `data/*.jsonl` file contains lines like:
```json
{"text": "force equals mass times acceleration", "domain": "physics", "difficulty": "basic"}
```

Domains: physics, chemistry, math, biology (~50 examples each).
To add data: create new .jsonl files in `data/` following the same format.

## Tech

- Python 3.12+
- PyTorch 2.x with CUDA
- Falls back to CPU + float32 if no GPU

## Git

- `main` branch, simple commits
- Commit format: `<type>: <what and why>`
- Types: feat, fix, refactor, docs

## Scaling Roadmap

- **1M params** (current): Character patterns, common words
- **10M params**: Grammar, sentence structure
- **100M+ params**: Knowledge, reasoning, fluency
