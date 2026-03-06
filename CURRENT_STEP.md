# Current Step: Run All Baseline Benchmarks (40 across 4 tiers)

**Status: IN PROGRESS -- GPQA Diamond done, 39 remaining**

**Previous: Setup llama-server + lm-eval-harness (DONE)**

## Why This Step

Every training phase compares against these baseline numbers. Without baselines, we can't measure improvement. All benchmarks must be scored at deployment quant (Q4_K_M) before any training begins. Even saturated benchmarks establish a reference point.

## Infrastructure (DONE)

- [x] llama-server b8215 CUDA 13.1 at `C:\AI\llama-cpp-server\` (85-92 t/s)
- [x] lm-evaluation-harness v0.4.11 in `.venv-eval/`
- [x] HuggingFace auth configured
- [x] `results/baseline/` directory created
- [x] Pipeline verified end-to-end (IFEval 2-sample smoke test)
- [x] Optimized server config: 6 parallel, 18K/slot, ~10.4GB VRAM (see CLAUDE.md)

## Completed Baselines

| Benchmark | Score | Metric | Stock FP16 | Date |
|-----------|-------|--------|------------|------|
| GPQA Diamond | **52.0%** | flexible-extract (generative CoT) | 81.7 | 2026-03-06 |

Results at: `results/baseline/Qwen3.5-9B-Q4_K_M.gguf/`

Note: 52% vs 81.7% stock gap is expected -- stock score uses loglikelihood MCQ scoring, our eval is generative zero-shot CoT (model must write reasoning + produce answer in extractable format). This is a harder but more honest evaluation.

## Remaining Benchmarks

### Tier Original (from Sovereign 10) -- 9 remaining

lm-eval-harness (run via llama-server API):
- [ ] IFEval (541 Qs, `ifeval`, `max_gen_toks=2048`, stock: 91.5)
- [ ] AIME 2025 (30 Qs, `aime25`, `max_gen_toks=8192`, stock: ~40-55 est.)

Separate repos/tools:
- [ ] SuperGPQA (stock: 58.2) -- github.com/TIGER-AI-Lab/SuperGPQA
- [ ] MMLU-Pro (stock: 82.5) -- needs generative variant, not loglikelihood
- [ ] LiveCodeBench v6 (stock: 65.6) -- github.com/LiveCodeBench/LiveCodeBench
- [ ] BFCL-V4 (stock: 66.1) -- github.com/ShishirPatil/gorilla (function calling)
- [ ] TAU2-Bench (stock: 79.1) -- github.com/sierra-research/tau2-bench
- [ ] RULER (N/A) -- github.com/hsiehjackson/RULER (long-context at 4K/16K/64K/128K)
- [ ] LongBench v2 (stock: 55.2) -- github.com/THUDM/LongBench

### Tier S -- Frontier-Differentiating, Unsaturated (20 benchmarks)

| # | Benchmark | Domain | Top Score | Notes |
|---|-----------|--------|-----------|-------|
| 1 | FrontierMath (T1-4) | Math / reasoning | ~40% T1-3 | Hardest math benchmark. Unpublished problems |
| 2 | Humanity's Last Exam (HLE) | Broad expert knowledge | ~45% | 2500 Qs, 100+ subjects |
| 3 | SWE-bench Pro | Real-world coding | ~57% | Multi-lang, multi-file |
| 4 | GDPval-AA | Real-world work | 83% GPT-5.4 | 44 occupations, real deliverables |
| 5 | METR Time Horizons | Agentic task duration | Scaling metric | How long can AI work autonomously |
| 6 | ARC-AGI-2 / ARC-AGI-3 | Fluid intelligence | ~54% raw | Novel reasoning, efficiency-constrained |
| 7 | SciCode | Research coding | 12% top | 338 sub-tasks from 80 lab problems |
| 8 | SimpleBench | Common-sense reasoning | ~62% vs 84% human | Spatial, temporal, social reasoning |
| 9 | FinanceArena / FinanceQA | Professional finance | ~50% | Practitioner-designed, exact-match |
| 10 | StockBench | Agentic trading | Most fail buy-and-hold | Sequential decisions, contamination-free |
| 11 | Terminal-Bench 2.0 | Agentic terminal work | ~78% | Sysadmin, data processing, SW eng |
| 12 | ProofBench (Vals AI) | Formal math proofs | GPT 5.4 leads | Formally verified proofs |
| 13 | Finance Agent v1.1 (Vals AI) | Financial analyst agent | Sonnet 4.6 leads | Core analyst tasks |
| 14 | CorpFin v2 (Vals AI) | Credit agreement analysis | Kimi K2.5 leads | Long-context financial docs |
| 15 | Vibe Code Bench v1.1 (Vals AI) | Build web apps | GPT 5.4 leads | Design + code + architecture |
| 16 | IOI (Vals AI) | Competitive informatics | GPT 5.4 leads | Olympiad-level algorithmic |
| 17 | SimpleQA Verified | Factual accuracy | ~55% F1 | Parametric knowledge without tools |
| 18 | FACTS Benchmark (Google) | Factual grounding | ~68% top | Multi-faceted factual accuracy |
| 19 | Enterprise Ops Bench (IBM) | Enterprise workflows | ~63% top | Domain-specific operational reasoning |
| 20 | APEX-Agents (Mercor) | Professional deliverables | GPT-5.4 leads | Law, finance, long-horizon work |

- [ ] FrontierMath
- [ ] HLE
- [ ] SWE-bench Pro
- [ ] GDPval-AA
- [ ] METR Time Horizons
- [ ] ARC-AGI-2
- [ ] SciCode
- [ ] SimpleBench
- [ ] FinanceArena
- [ ] StockBench
- [ ] Terminal-Bench 2.0
- [ ] ProofBench
- [ ] Finance Agent v1.1
- [ ] CorpFin v2
- [ ] Vibe Code Bench v1.1
- [ ] IOI
- [ ] SimpleQA Verified
- [ ] FACTS Benchmark
- [ ] Enterprise Ops Bench
- [ ] APEX-Agents

### Tier A -- Strong Signal (5 benchmarks)

| # | Benchmark | Domain | Notes |
|---|-----------|--------|-------|
| 1 | OTIS Mock AIME | Competition math | Harder than AIME, by olympiad students |
| 2 | IFBench | Instruction following | Modern IFEval replacement (AA Index) |
| 3 | InvestorBench | Financial decision-making | Equities, crypto, ETFs (ACL 2025) |
| 4 | ITBench (IBM/Kaggle) | IT automation | Enterprise IT, ~58% top |
| 5 | MCPMark | MCP tool ecosystems | 127 tasks testing MCP usage |

- [ ] OTIS Mock AIME
- [ ] IFBench
- [ ] InvestorBench
- [ ] ITBench
- [ ] MCPMark

### Tier B -- Solid Specialized Signal (5 benchmarks)

| # | Benchmark | Domain | Notes |
|---|-----------|--------|-------|
| 1 | CaseLaw v2 (Vals AI) | Legal reasoning | Private Canadian court-case QA |
| 2 | LegalBench (Vals AI) | Legal tasks | 109 models tested |
| 3 | TaxEval v2 (Vals AI) | Tax knowledge | Relevant to Austrian KESt research |
| 4 | MedCode (Vals AI) | Medical billing | Healthcare domain |
| 5 | Poker Agent (Vals AI) | Strategic game-playing | Which model makes most money |

- [ ] CaseLaw v2
- [ ] LegalBench
- [ ] TaxEval v2
- [ ] MedCode
- [ ] Poker Agent

## How To Run (lm-eval benchmarks)

### 1. Start llama-server (optimized)

```bash
cd C:\AI\llama-cpp-server
./llama-server.exe \
  -m "C:/Users/djord/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf" \
  --port 8080 --n-gpu-layers 99 --ctx-size 110592 --parallel 6 \
  --flash-attn on --jinja --host 127.0.0.1 --reasoning-format none
```

### 2. Run each benchmark

```bash
cd C:\AI\NeuralNetZero
source .venv-eval/Scripts/activate

# IFEval (~45-75 min at 6 concurrent)
PYTHONIOENCODING=utf-8 python -m lm_eval run \
  --model local-chat-completions \
  --model_args "model=Qwen3.5-9B-Q4_K_M.gguf,base_url=http://localhost:8080/v1/chat/completions,num_concurrent=6,max_gen_toks=2048" \
  --tasks ifeval \
  --batch_size 1 --apply_chat_template \
  --output_path results/baseline/ --log_samples

# AIME 2025 (~30-60 min, needs long thinking)
PYTHONIOENCODING=utf-8 python -m lm_eval run \
  --model local-chat-completions \
  --model_args "model=Qwen3.5-9B-Q4_K_M.gguf,base_url=http://localhost:8080/v1/chat/completions,num_concurrent=6,max_gen_toks=8192" \
  --tasks aime25 \
  --batch_size 1 --apply_chat_template \
  --output_path results/baseline/ --log_samples
```

### 3. Monitor from PowerShell

```powershell
# Watch progress (when running via Claude Code):
Get-Content "C:\Users\djord\AppData\Local\Temp\claude\C--AI-NeuralNetZero\tasks\<task_id>.output" -Wait

# Check server slots are active:
curl http://localhost:8080/slots

# Check VRAM:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

### Known Issues
- `PYTHONIOENCODING=utf-8` -- Windows cp1252 can't print Unicode in results table
- `--apply_chat_template` -- required for `local-chat-completions`
- `--reasoning-format none` -- keeps thinking in `content` for lm-eval
- GPQA is gated -- HF auth already configured
- Generative CoT scores are lower than loglikelihood MCQ -- this is expected and more honest

## After All Baselines

- [ ] Record 20 forgetting canary problems (5 knowledge, 5 math, 5 code, 5 instruction following)
- [ ] Compile all scores into `results/baseline/summary.json`
- [ ] Update PLAN.md with actual Q4_K_M baselines

## What Comes Next

Phase 1 (Vocab Surgery, optional) or Phase 2 (Domain-Adaptive CPT). Also: evaluate whether frontier API teachers (Claude Opus, GPT-5.3, Gemini 3.1 Pro via OpenRouter) can replace rented H100s for Phase 3 trace generation.
