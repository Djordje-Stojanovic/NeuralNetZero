# Sovereign Qwen 3.5 9B: The Plan
### Maximum local intelligence on RTX 5070 via sovereign post-training

---

## 1. Vision

Sovereign specialization of Qwen 3.5 9B -- a $30M+ frontier model, free to download -- into the most intelligent model running locally on a single RTX 5070. Instead of training from scratch, we specialize 9 billion parameters from "201 languages, all domains" to:

- **3 languages** (English, German, Serbian)
- **Pure intelligence** (reasoning, math, physics, code, tool use)
- **First-principles understanding** over broad trivia

CogCore-1B becomes **Track 2 (educational)**. Every insight from building a model from scratch feeds understanding of training dynamics. CogCore teaches. Sovereign Qwen delivers.

**Target: 15-50B dense equivalent on reasoning benchmarks.**

---

## 2. Base Model: Qwen 3.5 9B

### 2.1 Architecture

```
Model:          Qwen 3.5 9B (dense hybrid, NOT MoE)
Parameters:     ~9B total, ALL active
Layers:         32
Hidden dim:     4096
Vocab:          248,320 (padded)
Context:        262,144 native, extensible to 1,010,000 via YaRN
MTP:            Multi-Token Prediction heads (usable for speculative decoding)

Block layout (repeated 8x):
  3 x (Gated DeltaNet -> FFN)    [linear attention, O(n)]
  1 x (Gated Attention -> FFN)   [full attention, GQA: 16Q/4KV, head_dim=256]

Gated DeltaNet (24 layers):
  - Linear attention heads: 32 for V, 16 for QK
  - Head dimension: 128
  - Fixed-size state (no quadratic KV growth)
  - ~67M params per layer, ~1.62B total

Gated Attention (8 layers):
  - GQA: 16 query heads, 4 KV heads
  - Head dimension: 256
  - RoPE dim 64
  - ~59M params per layer, ~0.47B total

FFN (all 32 layers, SwiGLU):
  - Intermediate: 12288
  - ~151M params per layer, ~4.83B total

Embeddings + LM head (tied):
  - 248,320 x 4,096 = ~1.02B params (~11.3% of model)
```

### 2.2 Parameter Budget

| Component | Params | % of 9B |
|---|---|---|
| Embeddings + LM head (tied) | ~1.02B | 11.3% |
| 24x Gated DeltaNet attention | ~1.62B | 18.0% |
| 8x Gated Attention (GQA) | ~0.47B | 5.2% |
| 32x FFN (SwiGLU) | ~4.83B | 53.7% |
| Norms + misc | ~0.03B | 0.3% |
| **Total** | **~9B** | **100%** |

Key insight: FFNs hold 54% of capacity. The 8 Gated Attention layers are the global reasoning bottleneck -- highest-priority fine-tuning targets.

### 2.3 Why 9B Dense Over 35B-A3B MoE

| Factor | 9B Dense | 35B-A3B MoE |
|---|---|---|
| All weights in VRAM at Q4 | Yes (~5.3-6.5 GB) | No (needs CPU offload, ~20GB Q4) |
| Fine-tuning speed on 5070 | Fast (hours-days) | Very slow (weeks per stage) |
| GRPO RL feasibility | Comfortable | Painful, ~1 rollout at a time |
| Inference speed | 80-150 t/s at Q4 | 20-50 t/s with offloading |
| Post-training ceiling | 15-30B equivalent | 25-50B equivalent |

**Decision: Start with 9B.** Fast iteration > raw ceiling for pipeline development. 35B-A3B = future Track 3 after pipeline is proven.

---

## 3. Hardware

### 3.1 RTX 5070 Specs

```
GPU:              NVIDIA RTX 5070 (GB205, Blackwell)
VRAM:             12 GB GDDR7
Memory BW:        672 GB/s
FP16/BF16 Tensor: 61.7 TFLOPS (123.5 sparsity)
FP8 Tensor:       246.9 TFLOPS (493.9 sparsity)
MXFP4 native:     Yes (Blackwell microscaling FP4)
TDP:              250W
PCIe:             5.0 x16
FA support:       FA2 only (FA3 requires Hopper sm_90; Blackwell consumer = sm_120)
torch.compile:    Available on WSL2/Linux (10-30% speedup via Triton)
                  NOT available on native Windows
```

### 3.2 Inference Profile (Qwen 3.5 9B)

| Quantization | VRAM (model) | Max Context | Speed (t/s) | Quality |
|---|---|---|---|---|
| MXFP4 (native) | ~4.5-5.5 GB | 128K+ | 100-160 | ~ Q4_K_M or better |
| Q4_K_M (GGUF) | ~5.3-6.5 GB | 128K+ | 80-150 | Near-lossless |
| Q5_K_M | ~6-7.5 GB | 64-128K | 70-130 | Excellent |
| Q6_K | ~6.9-8 GB | 64K | 60-110 | Almost lossless |
| Q8_0 | ~8.9-10 GB | 32-64K | 50-90 | Overkill |

**Optimal:** MXFP4 or Q4_K_M with KV cache quantization (q8_0). Hybrid DeltaNet architecture keeps KV cache tiny (24/32 layers use fixed-size state).

### 3.3 Training Profile (QLoRA on 5070)

```
Framework:        Unsloth (native Qwen 3.5 support, Blackwell optimized)
Base model:       NF4 quantized (~5-6 GB)
LoRA rank:        16-32 (r=64 pushes VRAM limit)
LoRA targets:     all-linear (DeltaNet + GQA + FFN)
Max FT seq_len:   4096-8192 tokens (with gradient checkpointing)
Throughput:       ~200-800 tokens/sec effective
Optimizer:        8-bit AdamW or paged AdamW
Gradient ckpt:    Enabled (every layer)
```

### 3.4 Cloud Rental (8xH100)

```
Provider:       Lambda, RunPod, or PrimeIntellect
Node:           8xH100 80GB
Hourly:         ~$24/hr
Total rental:   ~20-40 hours across 2-3 sessions
Software:       vLLM (serving teachers), Axolotl/TRL (CPT)
```

---

## 4. The Pipeline: 12-Phase Sovereign Post-Training

### 4.0 Overview

```
Qwen 3.5 9B (stock)
    |
    v
Phase 1: Vocabulary Surgery (optional)
    |
    v
Phase 2: Domain-Adaptive Continued Pretraining [8xH100 rental]
    |
    v
Phase 3: Teacher Trace Generation [8xH100 rental]
    |
    v
Phase 4: Cold-Start Reasoning SFT [RTX 5070]
    |
    v
Phase 5: CoT Trajectory Distillation [RTX 5070]
    |
    v
Phase 6: On-Policy Knowledge Distillation (GKD) [RTX 5070]
    |
    v
Phase 7: GRPO Reasoning RL [RTX 5070 + optional H100 rental]
    |
    v
Phase 8: Multi-Stage Specialized RL [RTX 5070]
    |
    v
Phase 9: Tool Use + Agentic Training [RTX 5070]
    |
    v
Phase 10: Extended Thinking / Budget Forcing [RTX 5070]
    |
    v
Phase 11: Working Memory Enhancement [RTX 5070]
    |
    v
Phase 12: Merge + Quantize + Deploy [RTX 5070]
    |
    v
CogCore-9B-Sovereign v1
```

### 4.1 Phase 1: Vocabulary Surgery (Optional, Days 1-3)

**Goal:** Reduce 248K vocab to ~60-80K covering EN/DE/SR only. Free ~0.5-0.8B effective parameters.

**Method:**
1. Analyze tokenizer -- identify tokens exclusive to non-target languages (Chinese, Japanese, Korean, Arabic, Thai, Hindi, etc.)
2. Build pruned tokenizer: keep EN + DE + SR (Cyrillic + Latin) + code + special + math/science symbols
3. Use WECHSEL for embedding transfer (preserves learned representations for kept tokens)
4. Resize embedding matrix and LM head: 248,320 -> ~65,000-80,000
5. Short recovery SFT (1-2K steps) to stabilize
6. Verify: roundtrip encoding, no broken special tokens (`<|im_start|>`, `<think>`, etc.)

**Decision:** OPTIONAL. Skip for faster pipeline iteration and rely on CPT + SFT for language concentration.

### 4.2 Phase 2: Domain-Adaptive Continued Pretraining (CPT)

**Where:** Rented 8xH100 node ($200-400, 8-16 hours)
**Goal:** Rewire knowledge from "201 languages, all domains" to "3 languages, deep STEM/code/reasoning"

**Pre-filter: Base model as corpus scorer (RHO-1 style)**
Run stock Qwen 3.5 9B perplexity scoring on candidate corpus. Keep only documents where perplexity indicates the model has room to learn (high excess loss = high value). This produces 2-5x more efficient CPT by filtering out content the model already knows well.

**Corpus (target: 10-20B tokens):**

| Domain | Source | Tokens | % |
|---|---|---|---|
| English scientific papers | arXiv (physics, math, CS, engineering) | 4-6B | 30% |
| English code | The Stack v2 (filtered, high quality) | 3-5B | 25% |
| Mathematics | ProofPile-2 + OpenWebMath | 2-3B | 15% |
| German text | Wikipedia, news, technical docs, textbooks | 1-2B | 10% |
| Serbian text | Wikipedia, news, Cyrillic + Latin corpora | 0.5-1B | 5% |
| General English (replay) | FineWeb-Edu sample (forgetting prevention) | 1-2B | 10% |
| Physics textbooks | Feynman, Griffiths, Landau-Lifshitz (converted) | 0.5B | 5% |

**Training config:**
```
Method:         Full-parameter CPT (8xH100 can handle it)
LR:             1e-5 to 5e-5 (low, to not destroy base capabilities)
Schedule:       WSD (warmup 2%, stable 80%, decay 18%)
Batch:          Large (effective 2M+ tokens per step)
Context:        4096-8192
Epochs:         1 (single pass over curated corpus)
Replay:         10% general English mixed into every batch
```

### 4.3 Phase 3: Teacher Trace Generation

**Where:** Rented 8xH100 node ($200-500, 8-20 hours)
**Goal:** Generate 500K-1M verified reasoning traces from frontier teachers

**Teachers:**
- **DeepSeek-R1-671B** (full MoE): strongest open reasoning model
- **Qwen 3.5-72B** (or 122B-A10B): same architecture family, cleaner transfer
- Run BOTH, keep the better trace per problem

**Trace prompting strategy:**
```
"Derive the following result strictly from first principles.
Cite every axiom or law by name and equation number at the moment you use it.
After each step, verify consistency with prior results.
If any step fails verification, backtrack and correct.
Show your complete chain of reasoning.
Final answer in \boxed{} format."
```

**Target: ~400-600K verified traces across:**
Physics (50-75K), Mathematics (100-150K), Code (70-100K), Logic (35-50K), Tool use (35-50K), General intelligence (35-50K), First-principles physics (35-50K), SAP/ABAP (15-25K), Financial analysis (15-25K), Working memory tasks (15-25K)

**Rejection sampling:** Correctness, chain quality (axiom citations, self-verification), readability, diversity (cluster traces), difficulty (oversample hard problems).

### 4.4 Phase 4: Cold-Start Reasoning SFT

**Where:** RTX 5070, local (3-5 days)
**Goal:** Establish specific reasoning style before heavy distillation

**Data:** 10K highest-quality long-CoT examples from Phase 3 traces
- Prioritize: explicit axiom citations, self-verification, backtracking on errors
- Format: `<think>...</think>` matching Qwen 3.5's native format

**Training config:**
```
Framework:      Unsloth
Method:         QLoRA (NF4 base, LoRA r=32, alpha=64)
Targets:        all-linear (DeltaNet + GQA + FFN)
LR:             2e-4 (LoRA params only)
Schedule:       Cosine with warmup (5%)
Seq length:     4096-8192
Batch:          1-2 (gradient accumulation to effective 8)
Optimizer:      8-bit paged AdamW
Epochs:         2-3
DoRA:           use_dora=True (weight-decomposed LoRA, +0.5-1.5pp)
NEFTune:        neftune_noise_alpha=5 (+2-5pp instruction following)
rsLoRA:         use_rslora=True (stable at r=32-64)
```

### 4.5 Phase 5: CoT Trajectory Distillation (SFT)

**Where:** RTX 5070, local (7-14 days)
**Goal:** Absorb reasoning patterns from 671B/72B teachers. **Highest-impact phase** -- CoT distillation accounts for 60-80% of all post-training gains.

**Data:** Full 400-600K teacher traces from Phase 3

**Training config:**
```
Framework:      Unsloth
Method:         QLoRA (continue from Phase 4 LoRA weights)
LR:             1e-4 (slightly lower than cold-start)
Schedule:       Cosine with warmup (2%)
Batch:          2 (gradient accumulation to effective 16)
Epochs:         1 (data volume is large enough)
DoRA:           use_dora=True
NEFTune:        neftune_noise_alpha=5
rsLoRA:         use_rslora=True
Packing:        Enabled with sample-level attention masks (correctness fix)
Progressive seq: Start 2048, ramp to 4096 at 30% (20-30% faster early training)
Checkpoint avg: Average last 3-5 checkpoints at end (+0.5-1pp)
```

### 4.6 Phase 6: On-Policy Knowledge Distillation (GKD)

**Where:** RTX 5070, local (5-10 days)
**Goal:** Close train-inference gap from off-policy SFT. 4-8x more token-efficient than pure SFT.

**Method:**
1. Student (our fine-tuned 9B) generates solutions to problems
2. Score against teacher traces (KL divergence, correctness, chain quality)
3. Train on student's OWN generations close to teacher quality
4. Iterate: generate -> score -> train -> generate

```
Data:           Same problem set as Phase 3 (student generates fresh traces)
Rollouts:       4-8 per problem (limited by VRAM)
Loss:           Reverse KL (mode-seeking)
LR:             5e-5
Iterations:     3-5 rounds of generate->train
Checkpoint avg: Average last 3-5 checkpoints per round
```

### 4.7 Phase 7: GRPO Reasoning RL

**Where:** RTX 5070 local + optional 8xH100 rental ($100-200, 1-2 days)
**Duration:** 14-21 days local, or 1-2 days rented
**Goal:** Develop genuine reasoning strategies beyond teacher imitation

**Problem sources (40K ultra-hard prompts):**

| Tier | Source | Count | Reward Signal |
|---|---|---|---|
| Competition math | AIME, Putnam, IMO, FrontierMath-style | 10K | Binary correctness + verification |
| Advanced physics | First-principles derivation problems | 5K | Correctness + axiom-citation |
| Hard code | Codeforces Div 1+2, AtCoder, CodeContests | 10K | Execution (sandboxed) + efficiency |
| Formal math | Lean4 / Isabelle proof verification | 5K | Formal verifier (binary) |
| Complex tool use | Multi-turn agentic scenarios | 5K | Execution feedback |
| Logic / analysis | Graduate-level + multi-hop | 5K | LLM-as-judge + correctness |

**GRPO config:**
```
Rollouts per prompt:    8 (sequential on 5070, batched on H100)
Reward:                 Outcome-based (binary) + optional PRM step scoring
No critic model:        GRPO uses group relative baseline
LR:                     1e-6 to 5e-6
KL penalty:             0.01-0.05
Format reward:          +0.1 for proper \boxed{} or tool-call JSON
Stages:                 3 rounds of increasing difficulty
KV cache quant:         q8_0 during rollout generation (2x more rollouts on 12GB)
Freeze embeddings:      True (10-15% VRAM savings, embeddings stable by RL phase)
```

**VRAM budget (RTX 5070, 12GB):**
```
Base model (NF4):       ~5.0 GB
LoRA adapters (r=32):   ~0.3 GB
KV cache (q8_0, 4096):  ~0.8 GB
Rollout buffer (8x):    ~2.5 GB
Optimizer states:        ~1.5 GB
CUDA overhead:          ~1.5 GB
Total:                  ~11.6 GB (fits with sequential rollouts)
```

### 4.8 Phase 8: Multi-Stage Specialized RL

**Where:** RTX 5070, local (7-14 days total)

**Sub-stage 8a: Coding RL (5-7 days)**
- Problems: competitive programming + real-world code tasks (5K-10K)
- Reward: sandboxed execution + test case pass@1 + efficiency
- Expected gain: LiveCodeBench +5-10 points

**Sub-stage 8b: Agentic RL (3-5 days)**
- Problems: multi-turn tool use scenarios with execution (5K)
- Reward: trajectory success + tool-call correctness + efficiency
- Expected gain: BFCL/TAU2 +5-15 points

**Sub-stage 8c: Alignment RL via SimPO (2-3 days)**
- Method: SimPO (Simple Preference Optimization) -- no reference model needed, simpler than Constitutional AI
- Principles: truth-seeking, logical consistency, helpfulness, refuse bad financial advice
- Key: preserve reasoning power -- never penalize correct-but-blunt answers
- Data: 5K alignment preference pairs
- Expected gain: maintain benchmarks while adding safety

### 4.9 Phase 9: Tool Use + Agentic Training (SFT)

**Where:** RTX 5070, local (5-7 days)

**Data (50K-100K examples):** JSON/schema/function calling, multi-turn tool chains, error handling/retry, MCP protocol, code execution, file system ops, web search + synthesis

**Config:**
```
Method:         QLoRA (continue from merged RL weights, or new LoRA adapter)
LR:             1e-4
Seq length:     8192 (tool calls need long context)
Epochs:         2-3
DoRA:           use_dora=True
rsLoRA:         use_rslora=True
```

### 4.10 Phase 10: Extended Thinking / Budget Forcing

**Where:** RTX 5070, local (3-5 days)
**Goal:** Dynamically allocate thinking effort -- brief for easy, deep for hard.

**Method:**
1. During RL (Phase 7-8), reward longer chains ONLY when they lead to correct answers on hard problems
2. Penalize unnecessarily long chains on easy problems
3. Train on variable-length thinking examples (50-200 / 200-800 / 800-4000+ tokens)

**Budget forcing at inference:** "Think for at least N tokens" on hard problems. Published results show this pushes 9B into 30-50B territory on individual hard problems. +7-30% absolute on AIME/GPQA with scaled thinking tokens.

### 4.11 Phase 11: Working Memory Enhancement

**Where:** RTX 5070, local (3-5 days)
**Goal:** Exploit Gated DeltaNet layers' efficient state tracking.

**Training data (mix into SFT, 10-20% of batches, ~23K total):**
Variable binding (5K), State tracking (5K), Multi-hop reasoning (5K), Planning (3K), Code state machines (3K), Physics simulation (2K)

These "memory gym" tasks specifically exercise DeltaNet long-range state tracking.

### 4.12 Phase 12: Merge + Quantize + Deploy

**Where:** RTX 5070, local (1-2 days)

**Pipeline:**
1. Merge all LoRA adapters (PEFT/mergekit) -- TIES/DARE/SLERP for multiple adapters
2. Convert to GGUF: `convert-hf-to-gguf.py` (latest llama.cpp with Qwen 3.5 support)
3. Run imatrix calibration (include thinking tokens, tool calls, physics derivations, code)
4. Quantize: MXFP4 (optimal for Blackwell) or Q4_K_M (widest compatibility)
5. Benchmark against stock Qwen 3.5 9B
6. Checkpoint averaging: average last 3-5 checkpoints before final merge

**Deployment stack:**
```
Inference:      llama.cpp (sm_120 build) or vLLM; test ExLlamaV3 vs llama.cpp
Quantization:   MXFP4 or Q4_K_M with KV cache q8_0
MTP:            Enabled (speculative decoding, 1.5-2.5x speedup)
Prompt cache:   Enabled (system prompt + tool definitions cached)
Context:        64-128K practical (hybrid arch keeps KV tiny)
Speed:          80-150 t/s generation
Integration:    Continue.dev (VS Code/Neovim) + Ollama server
torch.compile:  Enable for serving on WSL2/Linux (10-30% speedup)
```

---

## 5. Domain-Specific Data Curation

Where YOUR 200-400 hours of work creates the real edge nobody else has.

### 5.1 First-Principles Physics Dataset

**Goal:** 10K-20K curated derivation chains where every step follows from axioms.

**Coverage:** Classical mechanics (Newton -> Lagrangian -> Hamiltonian), Electromagnetism (Maxwell from Coulomb + SR), Thermodynamics (from stat mech axioms), Quantum mechanics (postulates -> Schrodinger -> spin -> hydrogen), Special/General relativity, Statistical mechanics, Semiconductor physics (band theory -> transistor -> scaling laws).

### 5.2 SAP/ABAP Expert Dataset

**Goal:** 5K-10K examples covering production-level ABAP.

**Coverage:** ABAP syntax + OO ABAP, internal tables/ALV/selection screens, module-specific (MM, FI, SD, PP), BAPIs/RFCs/IDocs/ALE, debugging/performance tuning, S/4HANA migration patterns, CDS views/AMDP/RAP.

### 5.3 Financial Analysis Dataset

**Goal:** 5K-10K analytical reasoning examples (educational, never advisory).

**Coverage:** Owner earnings (Buffett method), DCF models, competitive analysis, 100-bagger screening (SQGLP + Akre), margin of safety, moat analysis. Every example includes: "This is an analytical framework demonstration, not investment advice."

### 5.4 Semiconductor Deep Knowledge

**Goal:** 3K-5K deep technical examples.

**Coverage:** Transistor physics (MOSFET -> FinFET -> GAA from first principles), process nodes (TSMC 3nm/2nm, Intel 18A), lithography (EUV/High-NA EUV), ASML stack, memory technologies (DRAM/HBM/NAND), packaging (CoWoS/chiplets/3D stacking).

---

## 6. Budget & Timeline

### 6.1 Compute Costs

| Phase | Where | Duration | Cost |
|---|---|---|---|
| Phase 2: CPT | 8xH100 rental | 8-16 hours | $200-400 |
| Phase 3: Trace generation | 8xH100 rental | 8-20 hours | $200-500 |
| Phase 7 (optional): RL boost | 8xH100 rental | 1-2 days | $100-200 |
| Phases 4-12: All local | RTX 5070 | ~40-60 days | ~$40-60 electricity |
| **Total compute** | | | **$540-1,160** |

### 6.2 Human Time

| Activity | Hours |
|---|---|
| Physics dataset curation | 80-120 |
| SAP/ABAP dataset creation | 30-50 |
| Financial analysis dataset | 20-30 |
| Semiconductor dataset | 15-25 |
| Working memory task design | 10-15 |
| Pipeline debugging + iteration | 40-60 |
| Benchmarking + evaluation | 20-30 |
| **Total** | **215-330 hours** |

### 6.3 Timeline

| Week | Activity |
|---|---|
| 1 | Setup: Unsloth, data pipeline, benchmark baseline stock Qwen 3.5 9B |
| 2 | Phase 2-3: Rent H100s, run CPT + generate teacher traces |
| 3 | Phase 4: Cold-start SFT (3-5 days) |
| 3-4 | Phase 5: CoT distillation SFT (7-14 days) |
| 5 | Phase 6: On-policy KD (5-10 days) |
| 5-7 | Phase 7: GRPO reasoning RL (14-21 days, or 1-2 days rented) |
| 7-8 | Phase 8: Specialized RL (coding, agentic, alignment) |
| 9 | Phase 9: Tool use SFT |
| 9-10 | Phase 10-11: Extended thinking + working memory |
| 10 | Phase 12: Merge, quantize, deploy, benchmark |
| **Total** | **~10-12 weeks** |

Dataset curation (215-330 hours) runs in parallel throughout.

---

## 7. Expected Outcomes

### 7.1 Benchmark Targets -- The Sovereign 10

Primary benchmark suite. All 10 run before and after every major phase.

| # | Benchmark | Measures | Stock 9B | Target | Equivalent |
|---|---|---|---|---|---|
| 1 | **GPQA Diamond** | Scientific reasoning (PhD-level, 198 Qs) | 81.7 | 85-90 | 30-50B |
| 2 | **SuperGPQA** | Broad graduate reasoning (285 disciplines) | 58.2 | 65-72 | 25-40B |
| 3 | **MMLU-Pro** | Knowledge + reasoning (10-option, graduate) | 82.5 | 82-86 | 15-20B |
| 4 | **AIME 2025** | Competition math (novel, post-training-data) | ~40-55 est. | 55-70 | 30-50B |
| 5 | **LiveCodeBench v6** | Practical coding (continuous refresh, exec-verified) | 65.6 | 72-80 | 20-30B |
| 6 | **BFCL-V4** | Tool use + function calling (multi-turn) | 66.1 | 80-90 | 25-35B |
| 7 | **TAU2-Bench** | Multi-turn agentic reasoning (enterprise) | 79.1 | 85-92 | 30-50B |
| 8 | **RULER** | Working memory + context degradation (4K-128K) | N/A | Establish baseline | -- |
| 9 | **IFEval** | Instruction following + format adherence | 91.5 | 92-95 | 30B+ |
| 10 | **LongBench v2** | Long-context reasoning (8K-2M) | 55.2 | 60-68 | 20-30B |

**Dead benchmarks (do NOT use for differentiation):** GSM8K (~96-99%, trivial), MMLU original (~92%+, saturated/contaminated), HumanEval original (~92-95%, too simple). These prove nothing in 2026.

**Custom domains (no standard benchmark):**

| Domain | Target | Equivalent |
|---|---|---|
| Physics derivations | Exceptional (first-principles from axioms) | Unique |
| SAP/ABAP | Expert-level production code | Unique |
| Financial analysis | Framework-grade analytical reasoning | Unique |
| German STEM | Excellent | Best-in-class at 9B |
| Serbian | Good+ | Unique |

### 7.2 The Honest Ceiling

**What we CAN achieve (15-50B equivalent):**
- Math/STEM reasoning: 30-50B equivalent (GRPO + distillation + extended thinking)
- Coding: 20-30B equivalent (RL + execution feedback)
- Tool use/agents: 25-35B equivalent (agentic RL)
- Working memory: 25-40B equivalent (DeltaNet advantage + targeted training)
- Physics understanding: unique depth
- Extended thinking on hard problems: individual problems can rival 70B+

**What we CANNOT achieve (parameter-bound):**
- Broad world knowledge (MMLU trivia): stays at ~9B level
- Full multilingual mastery: stays at 3 languages (by design)
- Long-form coherent generation (5000+ tokens): 9B ceiling applies
- Matching GPT-5.4 / Claude Opus 4.6 / Gemini 3.1 Pro on everything: 100x+ the compute

### 7.3 Unique Advantages

1. CoT distillation from 671B teacher (stock 9B never saw R1 traces)
2. First-principles physics training (derivation-from-axioms depth)
3. GRPO on ultra-hard problems (verifiable-reward RL on competition math/code)
4. Domain expertise (SAP/ABAP, semiconductors, investment analysis) -- bespoke, cannot be downloaded
5. Extended thinking training (calibrated variable-depth)
6. Working memory training (exercises DeltaNet state tracking)
7. 3-language specialization (concentrated vs spread across 201)
8. Tool use RL with execution feedback

### 7.5 Evaluation Framework

#### Primary Framework: lm-evaluation-harness

Covers 95% of benchmarks. Install: `pip install "lm_eval[hf,vllm]"`. Serve model via llama.cpp or vLLM on port 8000, then point all benchmarks at `http://localhost:8000/v1`.

#### Secondary Benchmarks (Phase Boundaries Only)

| Benchmark | Measures | When |
|---|---|---|
| **ARC-AGI-2** | Novel abstract logical reasoning | After Phase 7, Phase 12 |
| **FrontierMath** | Expert-original math (<2-5% frontier solve) | After Phase 7, Phase 12 |
| **SWE-bench Verified** | Multi-file real-world code debugging | After Phase 8a |
| **ZebraLogic** | Pure logic grid puzzles | After Phase 7 |
| **HMMT Feb 25** | Competition math (stock: 83.2) | After Phase 7 |
| **OJBench** | Algorithmic coding (stock: 29.2) | After Phase 8a |
| **BABILong** | Distributed fact reasoning over long context | After Phase 11 |
| **MultiChallenge** | Complex multi-constraint instruction following | After Phase 9 |

#### Custom Evaluations

| Domain | Method | Sample Size |
|---|---|---|
| **First-principles physics** | LLM-as-judge (rubric 0-10: axiom ID, logical flow, math correctness, self-verification, completeness) + human spot-check 20% | 200 problems |
| **SAP/ABAP** | ABAP sandbox execution + manual review | 100 scenarios |
| **Financial analysis** | Rubric: framework correctness, assumptions, math | 100 scenarios |
| **Semiconductor** | Custom MCQ (post-2025 papers) + open-ended derivations | 150 problems |
| **German STEM** | Translated MGSM + Eisvogel subset + custom physics | 200 problems |
| **Serbian fluency** | Custom translation + reasoning + Serbian-LLM-eval | 100 problems |
| **Extended thinking** | Accuracy vs thinking budget curves (50/200/800/2000/4000 tokens) | 100 hard problems x 5 budgets |

#### Quick Eval (<30 min, 750 problems)

Run after every LoRA checkpoint to detect regressions within a phase.

| Benchmark | Subset | Problems | Time |
|---|---|---|---|
| GPQA Diamond | 50 hardest (pre-selected) | 50 | ~5 min |
| MMLU-Pro | Random sample | 100 | ~5 min |
| MATH-500 | Full | 500 | ~10 min |
| LiveCodeBench mini | Recent 100 | 100 | ~10 min |

#### Full Eval (~5-6 hours)

Run at phase boundaries. All 10 Sovereign benchmarks at full size + 50 custom physics problems.

#### Evaluation Schedule per Phase

| Phase | After | Run |
|---|---|---|
| Pre-pipeline | Download stock model | **Full eval** (baseline "before" numbers) |
| Phase 1 (vocab surgery) | If done | Quick eval (verify no regression) |
| Phase 2 (CPT) | Completed CPT | **Full eval** (domain gains vs general regression) |
| Phase 3 (trace gen) | N/A | No eval (data generation, not training) |
| Phase 4 (cold-start SFT) | After 10K examples | Quick eval + custom physics 20 problems |
| Phase 5 (CoT distillation) | Every 50K examples | Quick eval. **Full eval** at end. |
| Phase 6 (on-policy KD) | Each round | Quick eval. **Full eval** at final round. |
| Phase 7 (GRPO RL) | Each difficulty stage | Quick eval + AIME full. **Full eval** at end. |
| Phase 8a (coding RL) | End | Quick eval + LiveCodeBench full + SWE-bench |
| Phase 8b (agentic RL) | End | Quick eval + BFCL-V4 full + TAU2 full |
| Phase 8c (alignment) | End | Quick eval + safety regression check |
| Phase 9 (tool use SFT) | End | Quick eval + BFCL-V4 + TAU2 + MultiChallenge |
| Phase 10 (extended thinking) | End | Quick eval + AIME with budget forcing curves |
| Phase 11 (working memory) | End | Quick eval + RULER full + BABILong |
| Phase 12 (deploy) | Final merged model | **Full eval** + ALL secondary + ALL custom + Arena submission prep |

#### Thinking Mode Protocol

1. Always enable thinking mode (`<think>...</think>`) during evaluation
2. Do NOT strip thinking tokens before scoring
3. Record both accuracy AND average thinking tokens per problem
4. Compare stock vs fine-tuned at equal thinking budgets AND natural budget
5. Budget forcing test on hard problems: accuracy vs token budget curves

#### Statistical Rigor

- **McNemar test** (paired, same problems) for "Model A beats Model B" claims (500+ samples, p<0.05)
- **Wilson confidence intervals** (95%) on all benchmark scores; non-overlapping CIs required for improvement claims
- **Paired t-test** on per-sample scores for "improved after Phase N" (200+ samples)

#### Regression Red Lines (Stop Training)

| Metric | Red Line | Action |
|---|---|---|
| GPQA Diamond | Drops >3pp from previous phase best | Stop. Investigate. Restore checkpoint. |
| MMLU-Pro | Drops >4pp from stock baseline | Increase replay buffer to 30%. |
| LiveCodeBench | Drops >5pp from previous phase | Restore checkpoint. Check data contamination. |
| IFEval | Drops >3pp from stock (below 88%) | Critical: format adherence broken. Fix before continuing. |
| Custom physics | Qualitative degradation (judge scores drop) | Review training data quality. |
| Any benchmark | Drops >2pp from immediate prior phase | Warning. Log and monitor next phase closely. |

#### Forgetting Canaries

20 problems checked every phase (5 general knowledge, 5 math, 5 code, 5 instruction following). If canary accuracy drops >20% relative: catastrophic forgetting in progress, increase replay buffer.

#### Contamination Prevention

1. **Blocklist** exact benchmark problem texts from all 10 Sovereign benchmarks in training data
2. **n-gram matching** (8-gram) to catch paraphrased versions; SBERT similarity (threshold 0.92) as secondary filter
3. **Exclude** known benchmark datasets and their hosting pages from CPT corpus
4. **Custom physics eval set**: keep offline/private, add canary strings, never upload to cloud
5. **Paraphrase audit** before claiming improvement: paraphrase 50 problems, if accuracy drops >15pp = contamination

#### Quantization Rule

Always benchmark at deployment quantization (Q4_K_M or MXFP4). BF16 numbers look better but mean nothing for deployment. Run BF16 once at baseline for reference, then everything at Q4/MXFP4.

#### Final Report Card Template

After Phase 12, produce this exact table:

```
=== CogCore-9B-Sovereign v1 -- Final Evaluation Report ===
Date: [date]
Hardware: RTX 5070 12GB, Q4_K_M / MXFP4
Framework: lm-evaluation-harness v0.4.x + custom

PRIMARY SUITE (Sovereign 10):
| Benchmark         | Stock 9B | Sovereign | Delta  | 95% CI    |
|-------------------|----------|-----------|--------|-----------|
| GPQA Diamond      | 81.7%    |   ?.?%    | +?.?pp | [?.?-?.?] |
| SuperGPQA         | 58.2%    |   ?.?%    | +?.?pp | [?.?-?.?] |
| MMLU-Pro          | 82.5%    |   ?.?%    | +?.?pp | [?.?-?.?] |
| AIME 2025         | ~??%     |   ?.?%    | +?.?pp | [?.?-?.?] |
| LiveCodeBench v6  | 65.6%    |   ?.?%    | +?.?pp | [?.?-?.?] |
| BFCL-V4           | 66.1%    |   ?.?%    | +?.?pp | [?.?-?.?] |
| TAU2-Bench        | 79.1%    |   ?.?%    | +?.?pp | [?.?-?.?] |
| RULER (128K)      | N/A      |   ?.?%    | base   | [?.?-?.?] |
| IFEval            | 91.5%    |   ?.?%    | +?.?pp | [?.?-?.?] |
| LongBench v2      | 55.2%    |   ?.?%    | +?.?pp | [?.?-?.?] |

SECONDARY: ARC-AGI-2, FrontierMath, SWE-bench, ZebraLogic, HMMT, BABILong
CUSTOM: Physics (0-10 scale), SAP/ABAP, Financial, Semiconductor, German STEM, Serbian
EXTENDED THINKING: AIME budget forcing curves (200/800/2000/4000/unbounded tokens)
WORKING MEMORY: RULER degradation curve (4K/16K/64K/128K)
INFERENCE: Quant, speed, max context, MTP speedup
STATISTICAL: McNemar p-values, CIs, contamination audit
```

---

## 8. Phase Checklist

### Pre-Pipeline
- [ ] Download Qwen 3.5 9B (GGUF for inference, HF for training)
- [ ] Setup lm-evaluation-harness (`pip install "lm_eval[hf,vllm]"`)
- [ ] Setup model serving (llama.cpp server or LM Studio API on port 1234)
- [ ] Run Full Eval (Sovereign 10) -- establish baseline
- [ ] Record 20 forgetting canary problems
- [ ] Setup Unsloth + PEFT + TRL + llama.cpp (Blackwell build, sm_120)
- [ ] Verify QLoRA training works: small test run (100 examples, 5 minutes)
- [ ] Setup vLLM for local inference (rollout generation)
- [ ] Prepare cloud rental account (Lambda/RunPod)
- [ ] Verify torch.compile works on WSL2/Linux

### Phase 1: Vocab Surgery (OPTIONAL)
- [ ] Analyze tokenizer: count tokens per language
- [ ] Build pruned tokenizer (~65-80K) with WECHSEL embedding transfer
- [ ] Resize embeddings + LM head
- [ ] Recovery SFT (1-2K steps)
- [ ] Verify: roundtrip encoding, special tokens intact

### Phase 2: Domain-Adaptive CPT
- [ ] Run base model perplexity scoring on candidate corpus (RHO-1 style filter)
- [ ] Curate CPT corpus (10-20B tokens, domain-weighted)
- [ ] Rent 8xH100 node
- [ ] Run CPT (8-16 hours)
- [ ] Benchmark: compare to stock on target domains + general MMLU

### Phase 3: Teacher Trace Generation
- [ ] Prepare seed problems (120K across all domains)
- [ ] Serve DeepSeek-R1-671B + Qwen 3.5-72B on 8xH100
- [ ] Generate traces (8-16 per problem)
- [ ] Rejection sample: correctness, quality, diversity
- [ ] Final count: 400-600K verified traces
- [ ] Download traces to local storage

### Phase 4: Cold-Start SFT
- [ ] Select top 10K traces
- [ ] Enable DoRA + NEFTune + rsLoRA
- [ ] QLoRA SFT on 5070 (3-5 days)
- [ ] Quick eval + custom physics 20 problems

### Phase 5: CoT Distillation
- [ ] Full 400-600K trace SFT on 5070 (7-14 days)
- [ ] Enable sample-level attention masks for packing
- [ ] Enable progressive seq length (2048 -> 4096 at 30%)
- [ ] Quick eval every 50K examples
- [ ] Average last 3-5 checkpoints at end
- [ ] **Full eval** at end. Expect +3-8 points on reasoning.

### Phase 6: On-Policy KD
- [ ] Setup GKD pipeline (student generates -> compare to teacher)
- [ ] Run 3-5 rounds of on-policy distillation (5-10 days)
- [ ] Checkpoint averaging per round
- [ ] Benchmark: expect additional +2-5 points

### Phase 7: GRPO Reasoning RL
- [ ] Prepare ultra-hard problem set (40K)
- [ ] Setup GRPO trainer (TRL + vLLM rollout)
- [ ] Enable KV cache q8_0 for rollouts
- [ ] Freeze embeddings
- [ ] Run RL on 5070 (14-21 days) or rent H100s (1-2 days)
- [ ] Quick eval + AIME full each difficulty stage. **Full eval** at end.
- [ ] Monitor: KL divergence, reward curves, benchmark scores

### Phase 8: Specialized RL
- [ ] Coding RL with execution feedback (5-7 days)
- [ ] Quick eval + LiveCodeBench full + SWE-bench after 8a
- [ ] Agentic RL with tool execution (3-5 days)
- [ ] Quick eval + BFCL-V4 full + TAU2 full after 8b
- [ ] Alignment RL via SimPO (2-3 days)
- [ ] Quick eval + safety regression check after 8c

### Phase 9: Tool Use SFT
- [ ] Curate tool use dataset (50-100K examples)
- [ ] Enable DoRA + rsLoRA
- [ ] QLoRA SFT (5-7 days)
- [ ] Quick eval + BFCL-V4 + TAU2 + MultiChallenge

### Phase 10: Extended Thinking
- [ ] Integrate variable-depth thinking into RL rewards
- [ ] Train budget forcing (3-5 days)
- [ ] Test: easy->short think, hard->long think
- [ ] Quick eval + AIME with budget forcing curves

### Phase 11: Working Memory
- [ ] Generate synthetic working memory tasks (23K)
- [ ] SFT with 10-20% memory tasks mixed in (3-5 days)
- [ ] Quick eval + RULER full + BABILong

### Phase 12: Deploy
- [ ] Merge all LoRA adapters (TIES/DARE/SLERP)
- [ ] Checkpoint averaging before final merge
- [ ] Run imatrix calibration
- [ ] Quantize to MXFP4 GGUF (or Q4_K_M)
- [ ] Test ExLlamaV3 vs llama.cpp performance
- [ ] Deploy via llama.cpp + Ollama + Continue.dev
- [ ] Enable torch.compile for serving (WSL2/Linux)
- [ ] **Full eval** + ALL secondary + ALL custom + Arena submission prep
- [ ] Enable MTP speculative decoding
- [ ] Enable KV cache quantization (q8_0)
- [ ] Enable prompt caching for system prompts

---

## 9. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Catastrophic forgetting during specialization | General capabilities degraded | 10-30% replay buffer in every stage |
| RL instability at 9B (GRPO collapse) | Wasted compute | KL penalty, early stopping, progressive difficulty |
| Teacher trace quality varies | Bad training signal | Rejection sampling, diversity filtering, verification |
| Unsloth doesn't support latest Qwen 3.5 ops | Can't train | Fallback to HF PEFT + TRL (slower but works) |
| VRAM OOM during RL (multiple rollouts) | Can't do RL locally | Sequential rollouts + KV cache quant, or rent H100s |
| Vocab surgery breaks special tokens | Model produces garbage | Skip vocab surgery, rely on CPT instead |
| 10-12 week timeline slips | Delayed deployment | Cut optional phases (vocab surgery, working memory) |
| RTX 5070 thermal throttling (sustained) | Slower training | Good case airflow, optional undervolt |
| DoRA/NEFTune regression on math | Lower math scores | Monitor after each phase, disable if regression |
| torch.compile incompatible with DeltaNet ops | No speedup | Fallback to eager mode |
| Benchmark contamination in training data | Inflated scores | n-gram blocklist + paraphrase audit (see 7.5) |

---

## 10. Success Criteria

CogCore-9B-Sovereign succeeds when it:

1. **Beats stock Qwen 3.5 9B** on GPQA Diamond, HMMT, AIME, LiveCodeBench, BFCL by +5 points minimum on 3+ benchmarks
2. **Produces verifiable first-principles physics derivations** at graduate textbook quality
3. **Generates production-quality ABAP code** for standard SAP enterprise scenarios
4. **Runs locally on RTX 5070** at 80+ tokens/sec with 32K+ usable context
5. **Handles multi-turn tool calling** with >80% success rate on structured function calls
6. **Adapts thinking depth** to problem difficulty (short for easy, long for hard)
7. **Operates in English, German, and Serbian** with native-quality fluency

---

## 11. Model Naming

```
Base:               Qwen 3.5 9B
After CPT:          CogCore-9B-CPT
After distillation: CogCore-9B-Distill
After RL:           CogCore-9B-RL
Final deployed:     CogCore-9B-Sovereign v1
```

---

## 12. What NOT to Do

| Technique | Why Skip |
|-----------|----------|
| Train from scratch at 9B | Use post-training instead ($30M+ compute already baked in) |
| Full fine-tune on consumer GPU | QLoRA only (VRAM constraint) |
| Skip replay buffer | Catastrophic forgetting |
| Skip distillation | Highest-impact technique (60-80% of gains) |
| Constitutional AI | Use SimPO instead (simpler, no ref model, better validated) |
| Attempt 35B MoE before 9B pipeline proven | Fast iteration > raw ceiling |
| Trust "3-7 day" estimates without overhead | Use realistic numbers |
| NEFTune during RL phases | Noisy embeddings hurt RL signal |
| Assume FA3 on consumer Blackwell | sm_120 = FA2 only as of March 2026 |

---

## 13. Track 2 -- CogCore-1B (Educational)

Existing code preserved as educational project:
- `model.py`, `config.py`, `train.py`, `tokenizer.py`, `optim.py`, `inference.py`
- 1M param dense baseline (n_layer=5, d_model=128, SwiGLU, RoPE)
- BPE tokenizer (8192 vocab)
- NOT a deployment target
- Every insight feeds back into understanding Sovereign pipeline dynamics

---

## 14. What Remains for v2

- Apply proven pipeline to Qwen 3.5 35B-A3B (Track 3 -- higher ceiling, needs 8xH100)
- Pruning: reduce 9B to effective 6B while keeping 90%+ reasoning
- Self-play data bootstrapping (model generates own training data)
- Formal verification integration (Lean4 as RL reward)
- Extended context training (32K+ fine-tuning)
- Voice interface (Qwen 3.5 TTS/ASR variants)
- S-LoRA multi-adapter serving
- Spectrum layer selection (SNR analysis, selectively freeze low-SNR layers)
- ExLlamaV3 vs llama.cpp deployment comparison
- Qwen 3.5-1.7B as verifier for non-verifiable RL problems

---

## 15. Sources

**Base model:** Qwen 3.5 technical report (Alibaba, Feb 2026)
**Distillation:** DeepSeek-R1 (2025), MobileLLM-R1 (Meta Feb 2026)
**RL:** GRPO (DeepSeek), Dr. GRPO/DAPO (ByteDance 2503.14476), Kimi K2 (Moonshot AI 2026)
**Architecture:** Gated DeltaNet (Qwen 3.5), Mamba-3 (ICLR 2026), Hymba (NVIDIA 2025), Falcon-H1
**Training:** Unsloth, PEFT (HuggingFace), TRL (HuggingFace)
**Optimizations:** DoRA (arXiv:2402.09353), NEFTune (arXiv:2310.05914), rsLoRA (arXiv:2312.03732), SimPO (arXiv:2405.14734), Spectrum (arXiv:2406.06623)
**Data:** RHO-1 (perplexity filtering), CLIMB (NVIDIA 2025), MATES (NeurIPS 2024)
**Inference:** llama.cpp, vLLM, ExLlamaV3, Ollama
**Evaluation:** lm-evaluation-harness, LiveCodeBench, BFCL (Berkeley), TAU2-Bench, RULER, LongBench v2, WildBench, ARC-AGI-2
**Prior work:** nGPT (arXiv 2410.01131), NorMuon (arXiv:2510.05491), Polar Express (arXiv:2505.16932)

---

*Strategic pivot from CogCore-1B scratch-build to sovereign post-training of Qwen 3.5 9B. Every technique referenced is published and replicable with consumer hardware as of March 2026.*
