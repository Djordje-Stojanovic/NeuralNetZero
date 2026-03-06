# Current Step: Run All Free Baseline Benchmarks

**Status: IN PROGRESS -- Config validated, re-running all benchmarks (0/22 valid baselines)**

**Previous: Setup llama-server + lm-eval-harness (DONE)**

## Why This Step

Every training phase compares against these baseline numbers. Without baselines, we can't measure improvement. All free benchmarks scored at deployment quant (Q4_K_M) before training begins.

## Infrastructure (DONE)

- [x] llama-server b8215 CUDA 13.1 at `C:\AI\llama-cpp-server\` (85-92 t/s)
- [x] lm-evaluation-harness v0.4.11 in `.venv-eval/`
- [x] HuggingFace auth configured
- [x] `results/baseline/` directory created
- [x] Optimized server config: 6 parallel, 18K/slot, ~10.4GB VRAM

## Universal Benchmark Config (MUST use for ALL benchmarks)

**Server — Standard (6×18K, most benchmarks):**
```bash
cd C:\AI\llama-cpp-server
./llama-server.exe \
  -m "C:/Users/djord/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf" \
  --port 8080 --n-gpu-layers 99 --ctx-size 110592 --parallel 6 \
  --flash-attn on --jinja --host 127.0.0.1 \
  --reasoning-format deepseek --no-context-shift \
  --chat-template-kwargs '{"enable_thinking":true}'
```

**Server — Heavy (4×32K, AIME/LiveCodeBench):** `--ctx-size 131072 --parallel 4`
**Server — Ultra (2×64K, re-run failures):** `--ctx-size 131072 --parallel 2`
Same flags as above, only ctx-size and parallel change.

**lm-eval — Standard config (most benchmarks):**
```bash
cd C:\AI\NeuralNetZero && source .venv-eval/Scripts/activate
PYTHONIOENCODING=utf-8 python -m lm_eval run \
  --model local-chat-completions \
  --model_args "model=Qwen3.5-9B-Q4_K_M.gguf,base_url=http://localhost:8080/v1/chat/completions,num_concurrent=6,max_gen_toks=16384,timeout=1800" \
  --tasks <TASK_NAME> \
  --batch_size 1 --apply_chat_template \
  --gen_kwargs "temperature=1.0,top_p=0.95,do_sample=true,max_gen_toks=16384" \
  --output_path results/baseline/ --log_samples
```

**lm-eval — Heavy thinking config (AIME, LiveCodeBench):**
Restart server with `--parallel 4 --ctx-size 131072` (32K/slot), then:
```bash
PYTHONIOENCODING=utf-8 python -m lm_eval run \
  --model local-chat-completions \
  --model_args "model=Qwen3.5-9B-Q4_K_M.gguf,base_url=http://localhost:8080/v1/chat/completions,num_concurrent=4,max_gen_toks=32768,timeout=1800" \
  --tasks <TASK_NAME> \
  --batch_size 1 --apply_chat_template \
  --gen_kwargs "temperature=1.0,top_p=0.95,do_sample=true,max_gen_toks=32768" \
  --output_path results/baseline/ --log_samples
```

**Why this config:**
- `--chat-template-kwargs '{"enable_thinking":true}'`: **REQUIRED for 9B.** Without this, model prefills empty `<think></think>` and 50%+ of responses have empty content. This was the root cause of AIME 0% score.
- `--reasoning-format deepseek`: Separates thinking into `reasoning_content`, eval harness only sees clean `content`
- `temperature=1.0,top_p=0.95`: Qwen/Unsloth official optimal params. **Model breaks at temp=0** (infinite `<think>` loop)
- `max_gen_toks` in BOTH `--model_args` AND `--gen_kwargs`: gen_kwargs overrides task YAML defaults
- `timeout=1800`: **NEVER use default (300s).** Let the model work until it finishes. If wrong, rerun. Never kill mid-thought.
- `do_sample=true`: Required for non-greedy. Without it, lm-eval ignores temperature.
- For math benchmarks: use custom task YAML with `doc_to_text` requesting `\boxed{}` format (see `custom_tasks/aime/`)
- If content is empty with `finish_reason=length`: model ran out of tokens while thinking. Use bigger slot config.
- Results are non-deterministic (temp=1.0 sampling). For small-N benchmarks, run 2-3x and report mean±std.

## Invalid Baselines (wrong config — kept for reference only)

| # | Benchmark | Score | Config Issue | Date |
|---|-----------|-------|-------------|------|
| - | GPQA Diamond | 52.0% | temp=0, reasoning-format none (thinking in content, possible truncation) | 2026-03-06 |
| - | IFEval | 25.9% | temp=0, reasoning-format none (thinking leaked into eval, 0% on format constraints) | 2026-03-06 |

These used `--reasoning-format none` + temp=0. Results in `results/baseline/` from T18-53 (GPQA) and T21-09 (IFEval) are INVALID.

## Valid Baselines

None yet. All previous runs used broken configs. AIME 2025 ran with correct thinking params but missing `enable_thinking=true` — needs re-run.

**AIME 2025 preliminary (needs re-run with enable_thinking):**
- lm-eval reported: 0/30 (0%) — WRONG due to extraction + empty content
- Real score with manual extraction: 12/30 (40%)
- When model produces content: 12/13 correct (92.3%)
- 17/30 empty content due to missing `--chat-template-kwargs '{"enable_thinking":true}'`
- Results in `results/baseline/Qwen3.5-9B-Q4_K_M.gguf/samples_aime25_2026-03-06T23-50-06.435673.jsonl`

## Remaining: 22 Free Benchmarks in 3 Phases

### Phase 1: lm-eval-harness cluster (9 benchmarks)

All use the universal server + lm-eval config above. No per-benchmark tweaks needed.

| # | Benchmark | Task Name | Qs | Expected 9B | Time Est. |
|---|-----------|-----------|-----|-------------|-----------|
| 1 | GPQA Diamond | `gpqa_diamond_cot_zeroshot` | 198 | 70-82% | ~2-4 hrs |
| 2 | IFEval | `ifeval` | 541 | 75-91% | ~2-4 hrs |
| 3 | AIME 2025 | `aime25` | 30 | 20-35% | ~1-2 hrs |
| 4 | MMLU-Pro | `mmlu_pro` | 12K | 70-82% | ~8-16 hrs |
| 5 | SimpleBench | `simplebench` | small | 40-60% | ~30-60 min |
| 6 | SimpleQA Verified | `simpleqa` | 1K | 30-45% | ~2-4 hrs |
| 7 | IFBench | `ifbench` | TBD | 60%+ | ~1-2 hrs |
| 8 | RULER | `ruler` | varies | 60-80% | ~2-4 hrs |
| 9 | LongBench v2 | `longbench2` | 1K | 40-55% | ~4-8 hrs |

Time estimates are higher than before because thinking mode generates ~8-16K tokens per question.

- [ ] AIME 2025 (re-run needed — with enable_thinking + custom YAML for \boxed{} format)
- [ ] IFEval (541 Qs, 6 parallel, 16K tokens)
- [ ] GPQA Diamond (198 Qs, 6 parallel, 16K tokens)
- [ ] MMLU-Pro
- [ ] SimpleBench
- [ ] SimpleQA Verified
- [ ] IFBench
- [ ] RULER
- [ ] LongBench v2

### Phase 2: Custom script benchmarks (8 benchmarks, ~6-10 hrs)

Need separate repos cloned + adapted to hit our OpenAI API. Windows OK (no Docker).

All use universal config (temp=1.0, top_p=0.95, reasoning-format deepseek). Adapt each repo's eval script to hit our llama-server OpenAI API.

| # | Benchmark | Repo | Qs | Expected 9B | Time Est. |
|---|-----------|------|-----|-------------|-----------|
| 10 | SuperGPQA | github.com/SuperGPQA/SuperGPQA | 14K | 50-58% | ~8-16 hrs |
| 11 | LiveCodeBench v6 | github.com/LiveCodeBench/LiveCodeBench | 600-800 | 55-66% | ~4-8 hrs |
| 12 | BFCL-V4 | github.com/ShishirPatil/gorilla | 1-2K | 50-66% | ~2-4 hrs |
| 13 | FinanceArena | github.com/AfterQuery/FinanceQA | 100s | 30-45% | ~1-2 hrs |
| 14 | SciCode | github.com/scicode-bench/SciCode | ~100 | <10% | ~2-4 hrs |
| 15 | HLE | github.com/centerforaisafety/hle | 2.5-3K | <15% | ~8-16 hrs |
| 16 | OTIS Mock AIME | Custom math eval | small | 15-30% | ~1-2 hrs |
| 17 | FACTS Benchmark | Google (Kaggle/HF) | 3.5K+ | <40% | ~8-16 hrs |

Time estimates higher than original because thinking mode generates ~8-16K tokens per question.

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

ALL benchmarks use Qwen optimal params: **temp=1.0, top_p=0.95, do_sample=true**.

**DO NOT use temp=0.** Qwen 3.5 9B loops infinitely in `<think>` at greedy decoding. This was discovered during IFEval testing — model generated infinite "Thinking Process:" repetition at temp=0, never producing `</think>` or an answer.

Results are non-deterministic with sampling. For small-N benchmarks (AIME=30), run 2-3 times and report mean±std.

## Next Task: After IFEval completes, re-run GPQA Diamond

IFEval is running now with the correct config. After it finishes:
1. Record IFEval results, update this doc
2. Re-run GPQA Diamond with universal config (old 52% was broken — temp=0, reasoning in content)
3. Continue with AIME 2025, then rest of Phase 1

Use the universal server + lm-eval config documented above for ALL remaining benchmarks. No per-benchmark config changes needed.

## After All Baselines

- [ ] Compile all scores into `results/baseline/summary.json`
- [ ] Update PLAN.md with actual Q4_K_M baselines
- [ ] Record 20 forgetting canary problems

## What Comes Next

Phase 1 (Vocab Surgery, optional) or Phase 2 (Domain-Adaptive CPT).
