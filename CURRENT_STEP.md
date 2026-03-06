# Current Step: Run All Free Baseline Benchmarks

**Status: IN PROGRESS -- GPQA Diamond + IFEval done (2/22), 20 remaining**

**Previous: Setup llama-server + lm-eval-harness (DONE)**

## Why This Step

Every training phase compares against these baseline numbers. Without baselines, we can't measure improvement. All free benchmarks scored at deployment quant (Q4_K_M) before training begins.

## Infrastructure (DONE)

- [x] llama-server b8215 CUDA 13.1 at `C:\AI\llama-cpp-server\` (85-92 t/s)
- [x] lm-evaluation-harness v0.4.11 in `.venv-eval/`
- [x] HuggingFace auth configured
- [x] `results/baseline/` directory created
- [x] Optimized server config: 6 parallel, 18K/slot, ~10.4GB VRAM

## Completed Baselines

| # | Benchmark | Score | Metric | Date |
|---|-----------|-------|--------|------|
| 1 | GPQA Diamond | **52.0%** | flexible-extract (generative CoT) | 2026-03-06 |
| 2 | IFEval | **25.9%** | prompt_level_strict_acc (generative) | 2026-03-06 |

## Remaining: 21 Free Benchmarks in 3 Phases

### Phase 1: lm-eval-harness cluster (8 benchmarks, ~8-12 hrs)

All run with same llama-server + lm-eval setup. No new installation needed. Windows native.

| # | Benchmark | Task Name | Qs | max_gen_toks | Temp | Expected 9B | Time Est. |
|---|-----------|-----------|-----|-------------|------|-------------|-----------|
| 2 | IFEval | `ifeval` | 541 | 2048 | 0 | 65-75% | ~45 min |
| 3 | AIME 2025 | `aime25` | 30 | 8192 | 0.7* | 20-35% | ~30 min |
| 4 | MMLU-Pro | `mmlu_pro` | 12K | 4096 | 0 | 45-55% | ~2 hrs |
| 5 | SimpleBench | `simplebench` | small | 2048 | 0 | 40-60% | ~10 min |
| 6 | SimpleQA Verified | `simpleqa` | 1K | 2048 | 0 | 30-45% | ~30 min |
| 7 | IFBench | `ifbench` | TBD | 2048 | 0 | 60%+ | ~30 min |
| 8 | RULER | `ruler` | varies | 4096 | 0 | 60-80% | ~30-60 min |
| 9 | LongBench v2 | `longbench2` | 1K | 4096 | 0 | 40-55% | ~45-90 min |

*AIME: Official uses sampling + majority vote (temp=0.7, k=16-32). Run greedy first for baseline, then optionally re-run with sampling.

- [x] IFEval -- 25.9% (reasoning-in-content issue, see ifeval_analysis.json)
- [ ] AIME 2025
- [ ] MMLU-Pro
- [ ] SimpleBench
- [ ] SimpleQA Verified
- [ ] IFBench
- [ ] RULER
- [ ] LongBench v2

### Phase 2: Custom script benchmarks (8 benchmarks, ~6-10 hrs)

Need separate repos cloned + adapted to hit our OpenAI API. Windows OK (no Docker).

| # | Benchmark | Repo | Qs | Temp | Expected 9B | Time Est. |
|---|-----------|------|-----|------|-------------|-----------|
| 10 | SuperGPQA | github.com/SuperGPQA/SuperGPQA | 14K | 0 | 35-45% | ~2-4 hrs |
| 11 | LiveCodeBench v6 | github.com/LiveCodeBench/LiveCodeBench | 600-800 | 0.7* | 25-40% | ~30-60 min |
| 12 | BFCL-V4 | github.com/ShishirPatil/gorilla | 1-2K | 0 | 40-60% | ~20-40 min |
| 13 | FinanceArena | github.com/AfterQuery/FinanceQA | 100s | 0 | 30-45% | ~30 min |
| 14 | SciCode | github.com/scicode-bench/SciCode | ~100 | 0 | <10% | ~30-60 min |
| 15 | HLE | github.com/centerforaisafety/hle | 2.5-3K | 0 | <15% | ~2-4 hrs |
| 16 | OTIS Mock AIME | Custom math eval | small | 0.7* | 15-30% | ~20 min |
| 17 | FACTS Benchmark | Google (Kaggle/HF) | 3.5K+ | 0 | <40% | ~1-2 hrs |

*LiveCodeBench/OTIS: Official uses pass@k sampling. Greedy first, then optionally re-run.

- [ ] SuperGPQA
- [ ] LiveCodeBench v6
- [ ] BFCL-V4
- [ ] FinanceArena
- [ ] SciCode
- [ ] HLE
- [ ] OTIS Mock AIME
- [ ] FACTS Benchmark

### Phase 3: Docker/WSL2 agent benchmarks (5 benchmarks)

Need Docker Desktop + WSL2. Agentic multi-turn. Low scores expected for 9B but still establishes baseline.

| # | Benchmark | Repo | Expected 9B | Time Est. |
|---|-----------|------|-------------|-----------|
| 18 | TAU2-Bench | github.com/sierra-research/tau2-bench | <20% | ~1-2 hrs |
| 19 | Terminal-Bench 2.0 | github.com/harbor-framework/terminal-bench | <10% | ~2-4 hrs |
| 20 | StockBench | github.com/ChenYXxxx/stockbench | marginal | ~2-4 hrs |
| 21 | SWE-bench Pro | github.com/SWE-bench/SWE-bench | <10% | ~4-8 hrs |
| 22 | ARC-AGI-2 | github.com/fchollet/ARC-AGI | <10% | ~2-4 hrs |

- [ ] TAU2-Bench
- [ ] Terminal-Bench 2.0
- [ ] StockBench
- [ ] SWE-bench Pro
- [ ] ARC-AGI-2

## Skipped (NOT free -- gated, paid, or platform-only)

| Benchmark | Reason |
|-----------|--------|
| FrontierMath | Private (Epoch AI), gated access |
| GDPval-AA | Gated (OpenAI) |
| METR Time Horizons | Private |
| APEX-Agents (Mercor) | Needs platform submission |
| Enterprise Ops Bench (IBM) | Gated |
| MCPMark | No public repo yet |
| InvestorBench | No clear public harness |
| All 10 Vals AI benchmarks | Platform-only for official scores |

(ProofBench, Finance Agent v1.1, CorpFin v2, Vibe Code Bench, IOI, CaseLaw v2, LegalBench, TaxEval v2, MedCode, Poker Agent)

## Sampling Strategy

| Mode | When | Config |
|------|------|--------|
| **Greedy (single run)** | Most benchmarks | temp=0, single run, deterministic |
| **Sampling + majority vote** | AIME, LiveCodeBench, OTIS, SciCode | temp=0.7, k=16-32, take majority answer |

For sampling benchmarks: run greedy first (quick baseline), then re-run with pass@k for official comparison.

## Next Task: Run Phase 1 (lm-eval cluster)

Start llama-server (optimized config from CLAUDE.md), then run all 8 Phase 1 benchmarks sequentially. Estimated ~8-12 hours total.

## After All Baselines

- [ ] Compile all scores into `results/baseline/summary.json`
- [ ] Update PLAN.md with actual Q4_K_M baselines
- [ ] Record 20 forgetting canary problems

## What Comes Next

Phase 1 (Vocab Surgery, optional) or Phase 2 (Domain-Adaptive CPT).
