# NeuralNetZero

Best intelligence-per-parameter LLM trained on first-principles STEM knowledge.
Hardware: RTX 5070, 12GB GDDR7 VRAM, Blackwell sm_120.

## Document Hierarchy

1. `CLAUDE.md` -- This file. Project identity, conventions, current state.
2. `PLAN.md` -- Long-term roadmap. Architecture spec, data strategy, all phases. (CR3 merged in)
3. `CURRENT_STEP.md` -- Active step details, sub-tasks, acceptance criteria.
4. `LLM_Training_Pipeline_Research_Report.md` -- Reference research (read-only).

If documents contradict each other, lower-numbered docs win.

## Current State

- 1M param dense model DONE (v1, char-level tokenizer, ~997K params)
- Phase 1 (BPE Tokenizer) DONE: 8192 vocab, 5.0x compression
- Next milestone: CogCore-500M (see PLAN.md)
- Current step: Phase 2 -- Data Pipeline (see CURRENT_STEP.md)

## The Algorithm (decision sequence for every change)

1. Question the requirement -- does this create learning value NOW?
2. Delete -- remove unnecessary complexity first
3. Simplify -- fewer files, fewer abstractions
4. Accelerate -- optimize only validated bottlenecks
5. Automate -- only proven stable workflows

## Principles

- Ship working code, not clever code
- Every parameter must be explainable
- Train smaller models longer (Qwen3 lesson: 60K tokens/param)
- Dense beats MoE below 1B params for IQ/parameter
- No dependencies until we need them

## File Size Rule

If any `.py` file exceeds 600 lines, split it into focused modules or simplify. Exception: `v0_pure_python.py` (preserved as-is, historical artifact).

## Architecture -- Current 1M Dense Baseline

```
n_layer=5, d_model=128, n_head=4, d_head=32
FFN: SwiGLU, inner_dim=344 (2.67x ratio)
Positions: RoPE
Norm: Pre-RMSNorm
Weight tying: lm_head = tok_emb
Biases: None
Tokenizer: Character-level (vocab ~49) + BPE (vocab 8192)
Context: 512 tokens
Total: ~997K parameters
```

## Architecture -- Target CogCore-500M

```
32L (26 DiffTrans + 6 Mamba2), d=896, 14 heads, d_head=64
DiffTrans + MLA (d_latent=224), QK-Norm, nGPT unit norms
MoE SwiGLU: 8 experts top-2, expert_hidden=448, shared=224
SkipV1 (layer-1 V reuse, alpha init=0), no weight tying, no biases
BPE 8192, context 2048, RoPE theta=10000 + YaRN/NTK
~490M params total, ~350M active at inference
Training: 80B tokens (~16 days on RTX 5070)
Post-training: Dr. GRPO (stripped DAPO), cold-start mandatory, SLERP merge
```

## Project Structure

- `v0_pure_python.py` -- Original ~800 param pure Python GPT (preserved, exempt from 600-line rule)
- `config.py` -- Model + training hyperparameters
- `tokenizer.py` -- CharTokenizer + BPETokenizer
- `model.py` -- GPT nn.Module (RoPE, SwiGLU, RMSNorm)
- `optim.py` -- Muon optimizer implementation
- `train.py` -- Main training script (entry point)
- `inference.py` -- Interactive text generation
- `prepare_corpus.py` -- Build tokenizer training corpus
- `train_tokenizer.py` -- Train BPE tokenizer (HuggingFace tokenizers)
- `tokenizer/` -- BPE model output (`stem_bpe.json`)
- `data/*.jsonl` -- STEM dataset (~200 examples across 4 domains)

## Usage

```bash
python train.py            # Train (char tokenizer by default)
python inference.py        # Generate text interactively
python prepare_corpus.py   # Build tokenizer corpus
python train_tokenizer.py  # Train BPE tokenizer
python v0_pure_python.py   # Original pure Python version
```

## Dataset Format

```json
{"text": "force equals mass times acceleration", "domain": "physics", "difficulty": "basic"}
```

Domains: physics, chemistry, math, biology (~50 examples each).

## Tech

- Python 3.12+
- PyTorch 2.10+ with CUDA 12.8 (cu128 for Blackwell)
- HuggingFace `tokenizers` for BPE
- Falls back to CPU + float32 if no GPU

## Git

- `main` branch, simple commits
- Commit format: `<type>: <what and why>`
- Types: feat, fix, refactor, docs
- After a successful commit where all tests pass, always push to main
