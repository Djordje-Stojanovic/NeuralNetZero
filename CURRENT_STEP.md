# Current Step: Run All 10 Sovereign Baseline Benchmarks

**Status: IN PROGRESS -- infrastructure ready, benchmarks not yet run**

**Previous: Setup llama-server + lm-eval-harness (DONE)**

## Why This Step

Every training phase compares against these baseline numbers. Without baselines, we can't measure improvement. All 10 Sovereign benchmarks must be scored at deployment quant (Q4_K_M) before any training begins.

## Infrastructure (DONE)

- [x] llama-server b8215 CUDA 13.1 at `C:\AI\llama-cpp-server\` (85-92 t/s)
- [x] lm-evaluation-harness v0.4.11 in `.venv-eval/`
- [x] HuggingFace auth configured
- [x] `results/baseline/` directory created
- [x] Pipeline verified end-to-end (IFEval 2-sample smoke test)

## Benchmarks To Run

### Group A: lm-eval-harness (3 benchmarks, run via llama-server API)

These use the verified pipeline. Start server, then run each:

| # | Benchmark | Task Name | Questions | max_gen_toks | Est. Time |
|---|-----------|-----------|-----------|-------------|-----------|
| 1 | GPQA Diamond | `gpqa_diamond_cot_zeroshot` | 198 | 4096 | ~1-2 hrs |
| 9 | IFEval | `ifeval` | 541 | 2048 | ~2-4 hrs |
| 4 | AIME 2025 | `aime25` | 30 | 8192 | ~1-2 hrs |

- [ ] GPQA Diamond (stock target: 81.7)
- [ ] IFEval (stock target: 91.5)
- [ ] AIME 2025 (~40-55 est.)

### Group B: Separate repos/tools (7 benchmarks, need installation)

| # | Benchmark | Repo/Tool | Stock Target | Notes |
|---|-----------|-----------|-------------|-------|
| 2 | SuperGPQA | github.com/TIGER-AI-Lab/SuperGPQA | 58.2 | Separate eval script |
| 3 | MMLU-Pro | lm-eval with generative variant OR separate repo | 82.5 | Default loglikelihood won't work with chat API |
| 5 | LiveCodeBench v6 | github.com/LiveCodeBench/LiveCodeBench | 65.6 | Code execution sandbox needed |
| 6 | BFCL-V4 | github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard | 66.1 | Function calling eval |
| 7 | TAU2-Bench | github.com/sierra-research/tau2-bench | 79.1 | Agentic retail/airline tasks |
| 8 | RULER | github.com/hsiehjackson/RULER | N/A | Long-context, test at 4K/16K/64K/128K |
| 10 | LongBench v2 | github.com/THUDM/LongBench | 55.2 | Long-context comprehension |

- [ ] MMLU-Pro (stock target: 82.5)
- [ ] SuperGPQA (stock target: 58.2)
- [ ] LiveCodeBench v6 (stock target: 65.6)
- [ ] BFCL-V4 (stock target: 66.1)
- [ ] TAU2-Bench (stock target: 79.1)
- [ ] RULER (establish baseline at 4K/16K/64K/128K)
- [ ] LongBench v2 (stock target: 55.2)

## How To Run (Group A)

### 1. Start llama-server

```bash
cd C:\AI\llama-cpp-server
./llama-server.exe \
  -m "C:/Users/djord/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf" \
  --port 8080 --n-gpu-layers 99 --ctx-size 16384 --parallel 2 \
  --flash-attn on --jinja --host 127.0.0.1 --reasoning-format none
```

### 2. Run each benchmark

```bash
cd C:\AI\NeuralNetZero
source .venv-eval/Scripts/activate

# GPQA Diamond (~1-2 hrs)
PYTHONIOENCODING=utf-8 python -m lm_eval run \
  --model local-chat-completions \
  --model_args "model=Qwen3.5-9B-Q4_K_M.gguf,base_url=http://localhost:8080/v1/chat/completions,num_concurrent=2,max_gen_toks=4096" \
  --tasks gpqa_diamond_cot_zeroshot \
  --batch_size 1 --apply_chat_template \
  --output_path results/baseline/ --log_samples

# IFEval (~2-4 hrs)
PYTHONIOENCODING=utf-8 python -m lm_eval run \
  --model local-chat-completions \
  --model_args "model=Qwen3.5-9B-Q4_K_M.gguf,base_url=http://localhost:8080/v1/chat/completions,num_concurrent=2,max_gen_toks=2048" \
  --tasks ifeval \
  --batch_size 1 --apply_chat_template \
  --output_path results/baseline/ --log_samples

# AIME 2025 (~1-2 hrs)
PYTHONIOENCODING=utf-8 python -m lm_eval run \
  --model local-chat-completions \
  --model_args "model=Qwen3.5-9B-Q4_K_M.gguf,base_url=http://localhost:8080/v1/chat/completions,num_concurrent=2,max_gen_toks=8192" \
  --tasks aime25 \
  --batch_size 1 --apply_chat_template \
  --output_path results/baseline/ --log_samples
```

### Known Issues & Fixes
- `PYTHONIOENCODING=utf-8` -- Windows cp1252 can't print Unicode arrows in results table
- `--apply_chat_template` -- required for `local-chat-completions` model backend
- `--reasoning-format none` on server -- keeps thinking in `content` for lm-eval to see
- `--ctx-size 16384 --parallel 2` -- safe for 12GB VRAM. 8K context too small with thinking.
- GPQA is gated -- HF auth already configured

## After All 10 Benchmarks

- [ ] Record 20 forgetting canary problems (5 knowledge, 5 math, 5 code, 5 instruction following)
- [ ] Compile all scores into `results/baseline/summary.json`
- [ ] Update PLAN.md Sovereign 10 table with actual Q4_K_M baselines

## What Comes Next

Phase 1 (Vocab Surgery, optional) or Phase 2 (Domain-Adaptive CPT). Also: evaluate whether frontier API teachers (Claude Opus, GPT-5.3, Gemini 3.1 Pro via OpenRouter) can replace rented H100s for Phase 3 trace generation.
