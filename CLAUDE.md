# NeuralNetZero

Sovereign post-training of Qwen 3.5 9B for maximum local intelligence on RTX 5070.
Hardware: RTX 5070, 12GB GDDR7 VRAM, Blackwell sm_120.

## Document Hierarchy

1. `CLAUDE.md` -- This file. Project identity, conventions, current state.
2. `PLAN.md` -- Long-term roadmap. 12-phase Sovereign pipeline, architecture, data strategy. (Sovereign CR merged in)
3. `CURRENT_STEP.md` -- Active step details, sub-tasks, acceptance criteria. **Always update after finishing a task** (check off completed sub-tasks, update status, advance to next step if done).
4. `LLM_Training_Pipeline_Research_Report.md` -- Reference research (read-only).

If documents contradict each other, lower-numbered docs win.

## Current State

- **Track 1 (Sovereign Qwen 3.5 9B):** Strategic pivot -- post-train Qwen 3.5 9B into CogCore-9B-Sovereign
- Model downloaded + running at 75-85 t/s in LM Studio (Q4_K_M GGUF)
- Current step: Pre-Pipeline Setup -- baseline benchmarking in progress (see CURRENT_STEP.md)
- Evaluation framework: Sovereign 10 benchmarks defined (see PLAN.md 7.5)
- Target: 15-50B dense equivalent on reasoning, 80+ t/s local inference
- Pipeline: 12 phases, $540-1160, 10-12 weeks
- **Track 2 (CogCore-1B Educational):** Existing code preserved, not deployment target
- 1M param dense model DONE (v1, char-level tokenizer, ~997K params)
- BPE Tokenizer DONE (8192 vocab, 5.0x compression)

## The Algorithm (decision sequence for every change)

1. Question the requirement -- does this create learning value NOW?
2. Delete -- remove unnecessary complexity first
3. Simplify -- fewer files, fewer abstractions
4. Accelerate -- optimize only validated bottlenecks
5. Automate -- only proven stable workflows

## Principles

- Specialize a frontier model rather than train from scratch
- Fast iteration > raw ceiling (9B before 35B MoE)
- Ship working code, not clever code
- Every parameter must be explainable
- Dense beats MoE for single-GPU deployment
- No dependencies until we need them

## File Size Rule

If any `.py` file exceeds 600 lines, split it into focused modules or simplify. Exception: `v0_pure_python.py` (preserved as-is, historical artifact).

## Architecture -- Target CogCore-9B-Sovereign

```
Base:           Qwen 3.5 9B (dense hybrid, Gated DeltaNet + GQA)
Params:         ~9B total, ALL active
Layers:         32 (24x Gated DeltaNet + 8x Gated Attention w/ GQA 16Q/4KV)
Hidden:         4096, FFN intermediate 12288 (SwiGLU)
Vocab:          248,320, Context: 262K native
Pipeline:       12 phases (vocab surgery -> CPT -> traces -> SFT -> distillation -> GKD -> GRPO -> specialized RL -> tool SFT -> thinking -> memory -> deploy)
Training:       QLoRA on RTX 5070 + rented 8xH100 for CPT/traces
Target:         15-50B equivalent on reasoning, 80+ t/s local
Model name:     CogCore-9B-Sovereign v1
```

## Architecture -- Track 2: 1M Dense Baseline (Educational)

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

## Project Structure

### Track 2 (Educational, existing code)
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

### Track 1 (Sovereign Pipeline, TBA)
- Pipeline scripts for each phase will be added as implemented

## Usage

```bash
# Track 2 (Educational)
python train.py            # Train 1M model
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
- Unsloth (QLoRA training, Qwen 3.5 support)
- PEFT (LoRA/DoRA adapters)
- TRL (GRPOTrainer, SFTTrainer)
- llama.cpp (inference, sm_120 build)
- vLLM (batched inference for RL rollouts)
- lm-evaluation-harness (benchmark evaluation framework)
- Ollama (deployment/serving)
- torch.compile available on WSL2/Linux (NOT native Windows)
- FA2 only (FA3 requires Hopper sm_90, not available on sm_120)
- Falls back to CPU + float32 if no GPU

## Git

- `main` branch, simple commits
- Commit format: `<type>: <what and why>`
- Types: feat, fix, refactor, docs
- After a successful commit where all tests pass, always push to main
