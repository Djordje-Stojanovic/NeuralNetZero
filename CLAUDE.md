# NeuralNetZero

Train LLMs from scratch, starting at ~100 parameters and scaling up.
Learn every concept hands-on: autograd, backprop, attention, transformers.
Hardware: RTX 5070 Super (will use later when we scale to GPU).

## The Algorithm (decision sequence for every change)

1. Question the requirement — does this create learning value NOW?
2. Delete — remove unnecessary complexity first
3. Simplify — fewer files, fewer abstractions
4. Accelerate — optimize only validated bottlenecks
5. Automate — only proven stable workflows

## Principles

- Ship working code, not clever code
- One file per concept until complexity demands splitting
- Every parameter must be explainable
- Scale gradually: 100 -> 1K -> 10K -> 100K -> 1M+ params
- Pure Python first, PyTorch/CUDA later when we need speed
- No dependencies until we need them

## Project Structure

- `microgpt.py` — the complete algorithm (Karpathy-style, pure Python)
- Future: scaled versions with PyTorch, CUDA, larger datasets

## Tech

- Python 3.12+
- Pure stdlib (math, random, os) for initial implementation
- PyTorch + CUDA later for GPU scaling

## Git

- `main` branch, simple commits
- Commit format: `<type>: <what and why>`
- Types: feat, fix, refactor, docs
