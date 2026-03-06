# From-Scratch LLM Training Pipeline: Research Report
## 1K to 100M+ Parameters on a Single RTX 5070 (12GB VRAM)
### As of March 5, 2026

---

## Table of Contents

1. [Architecture Innovations for Micro-to-Small LLMs](#1-architecture-innovations)
2. [Training Efficiency and Optimization](#2-training-efficiency)
3. [Data Efficiency — Maximizing Learning per Token](#3-data-efficiency)
4. [Tokenization](#4-tokenization)
5. [Evaluation and Understanding What the Model Learns](#5-evaluation)
6. [Cutting-Edge Techniques (2025–2026)](#6-cutting-edge)
7. [Scale Milestone Recipes](#7-milestone-recipes)

---

## 1. Architecture Innovations

### 1.1 Component Choices That Matter at Small Scale

**Positional Encoding: RoPE wins.** At every scale from 10K to 100M parameters, Rotary Position Embeddings (RoPE) are the clear default. Sinusoidal embeddings waste parameters on a fixed scheme. ALiBi is simpler (no learned/computed embeddings, just a linear bias in attention) and works well for extrapolation, but RoPE gives better in-distribution quality and is what nanochat, LLaMA, Qwen, Phi, and essentially every modern architecture uses. At micro scale (<100K params), ALiBi is a reasonable alternative since it adds zero parameters. **Recommendation: RoPE at all scales ≥100K params. ALiBi acceptable below that.**

**Activation Function: SwiGLU is king.** SwiGLU (used in LLaMA, Qwen, Phi, nanochat) consistently outperforms ReLU and GELU in quality-per-FLOP. The tradeoff: SwiGLU uses a gated FFN with three weight matrices instead of two, so for the same hidden dimension the FFN has 50% more parameters. The standard fix is to reduce the FFN intermediate dimension — use a ratio of ~2.67× d_model instead of the classic 4× for non-gated FFNs. This keeps total parameter count comparable while getting better quality. At micro scale (<10K params) where every parameter matters, plain GELU works fine — the SwiGLU advantage is marginal when the FFN is tiny. **Recommendation: SwiGLU at ≥1M params, GELU below that.**

**Normalization: RMSNorm, pre-norm.** RMSNorm is ~15% faster than LayerNorm (no mean subtraction) with equivalent or better training stability. Pre-norm (normalize before attention/FFN, not after) is essential — it enables deeper networks without gradient issues. This is the universal standard in all modern architectures. **Recommendation: Pre-RMSNorm at all scales. No debate.**

**Weight Tying: Always at small scale.** Tying input and output embedding matrices saves V × d_model parameters. For a 32K vocab with d_model=256, that's 8M parameters — potentially more than the rest of the model at small scale. Every small model should use weight tying. nanochat uses it. LLaMA doesn't (at 7B+ it's a rounding error). **Recommendation: Always tie weights below 100M params.**

**Attention: Grouped Query Attention (GQA).** Multi-head attention (MHA) is the baseline. GQA reduces KV-head count (e.g., 8 query heads, 2 KV heads) saving parameters and memory with minimal quality loss. At micro scale (<1M params) you may only have 2–4 heads total, making GQA pointless. At 10M+ params with 8+ heads, GQA with a 4:1 ratio is a free lunch. Multi-Query Attention (MQA, single KV head) is too aggressive for training quality but excellent for inference. **Recommendation: MHA below 10M params, GQA (4:1 or 8:2 ratio) at 10M+.**

**Bias terms: Remove them.** Following LLaMA/Qwen/Phi, removing bias from all linear layers and norms simplifies the model, slightly reduces parameters, and often improves generalization. BitNet also removes biases. **Recommendation: No biases anywhere.**

### 1.2 Optimal Depth-to-Width Ratio by Scale

This is one of the most important architectural decisions. Key findings:

- Research by Levine et al. ("Which Transformer Architecture Fits My Data?") proved that **vocabulary size directly constrains the optimal depth-to-width ratio**. With character-level tokenization (vocab ~256), width beyond ~256 is wasted — the embedding matrix becomes rank-deficient. This is critical for your pipeline: when using character-level tokenization, go deep and narrow.
- The "Impact of Depth and Width on Transformer Language Model Generalization" paper found that deeper models generalize better for compositional tasks, but **returns diminish rapidly** and excessive depth hurts.
- nanochat's `--depth` parameter automatically computes width, heads, and all hyperparameters from depth alone, following empirical scaling recipes.

**Concrete recommendations:**

| Scale | Params | Layers | d_model | Heads | d_head | FFN dim | Notes |
|-------|--------|--------|---------|-------|--------|---------|-------|
| Micro | ~10K | 2 | 64 | 2 | 32 | 172 | Character-level only viable here |
| Small | ~100K | 4 | 128 | 4 | 32 | 344 | Start seeing coherent patterns |
| Medium | ~1M | 6 | 256 | 4 | 64 | 688 | First "real" language model |
| Large | ~10M | 12 | 384 | 6 | 64 | 1032 | Can learn grammar, basic facts |
| XL | ~100M | 16 | 768 | 12 | 64 | 2048 | GPT-2 Small territory |

**Key rules of thumb:**
- d_head should be 32–128 at all scales (empirical sweet spot for per-head expressiveness)
- d_model should never exceed vocab_size when using character-level tokenization (rank bottleneck)
- FFN ratio: ~2.67× d_model with SwiGLU, ~4× without
- Prefer slightly deeper over wider when in doubt — depth helps compositional generalization
- At 100M params, 16–24 layers with d_model 512–768 is the sweet spot

### 1.3 Mixture of Experts (MoE) at Small Scale

**Short answer: Not worth it below 100M total params.**

MoE shines when you want to decouple total parameters (capacity) from active parameters (compute per token). The Moonlight model (3B/16B MoE) demonstrates the benefit at scale. But MoE introduces:
- Router complexity and load-balancing losses
- Minimum viable expert count (typically 8+)
- Each expert needs enough parameters to be useful
- Auxiliary losses for balanced routing add hyperparameter complexity

At 10M params, if you split into 8 experts of ~1M active each, each expert is too small to specialize meaningfully. The routing overhead eats your gains.

**Exception:** At 50M–100M total params with top-2 of 8 routing, you could get an effective 25M active-parameter model with 100M capacity. This is on the edge of useful. If you're feeling adventurous at the 100M milestone, try it as an experiment — but expect dense models to be simpler and equally good.

---

## 2. Training Efficiency and Optimization

### 2.1 Optimizer: Muon vs AdamW

This is the biggest optimization story of 2025.

**Muon** (MomentUm Orthogonalized by Newton-schulz) was created by Keller Jordan for the nanoGPT speedrun competition and has since been validated at scale by Moonshot AI's Moonlight model (3B/16B MoE, 5.7T tokens). The key findings:

- **~2× compute efficiency over AdamW** — Muon reaches the same loss in half the training FLOPs
- Muon orthogonalizes the gradient momentum via Newton-Schulz iteration, treating weight matrices as geometric objects rather than flat parameter vectors
- **33% less optimizer memory** than AdamW (no second moment / v state)
- The FLOP overhead of Newton-Schulz iteration is <1% of total training compute
- Critical for scaling: must add weight decay (vanilla Muon without weight decay diverges in long training runs) and adjust per-parameter update scale

**Practical implementation for your pipeline:**

Muon is used **only for 2D weight matrices** (attention projections, FFN layers). Embeddings, layer norms, bias terms, and 1D parameters still use AdamW. This dual-optimizer setup is standard.

```python
# Pseudocode — Muon for matrix params, AdamW for the rest
matrix_params = [p for n, p in model.named_parameters() if p.dim() == 2 and 'embed' not in n]
other_params = [p for n, p in model.named_parameters() if p.dim() != 2 or 'embed' in n]

optimizer = CombinedOptimizer([
    Muon(matrix_params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.01),
    AdamW(other_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
])
```

**Important caveat:** If you pretrain with Muon, you should also finetune with Muon. Switching optimizers between pretraining and SFT causes degradation (confirmed by the Moonlight paper).

**At micro scale (<1M params):** AdamW is fine. Muon's advantage grows with matrix size, and tiny matrices don't benefit much from spectral orthogonalization. Use Muon starting at ~1M params.

**Turbo-Muon** (Boissin et al., Dec 2025) replaces the Newton-Schulz step with spectral preconditioning (Polar Express), giving 8–10% faster step times with identical quality. If available in your framework, use it.

### 2.2 Learning Rate Schedules

**Warmup-Stable-Decay (WSD) is the pragmatic choice.**

| Schedule | Pros | Cons | Best For |
|----------|------|------|----------|
| Cosine | Well-studied, smooth | Must know total steps upfront; can't resume/extend | Fixed-length runs |
| Linear decay | Simple | Slightly worse than cosine empirically | Quick experiments |
| WSD (trapezoidal) | Can extend training; stable phase is easy to resume | Less studied at small scale | Production training |
| Cosine w/ restarts | Helps escape local minima | Complex, more hyperparams | Multi-phase training |

nanochat uses cosine with linear warmup. For your single-GPU pipeline where you might want to train longer if things look promising, WSD is more practical. The "trapezoidal" schedule from recent work (warmup → hold → decay) is essentially WSD.

**Concrete numbers:**
- Warmup: 1–5% of total steps (e.g., 200–2000 steps)
- Peak LR with Muon for matrices: 0.01–0.05 (Muon uses much higher LRs than AdamW)
- Peak LR with AdamW for other params: 1e-4 to 6e-4
- Final LR: 10% of peak (0.1× multiplier)
- Weight decay: 0.01 for Muon params, 0.1 for AdamW params

### 2.3 Gradient Accumulation on Single GPU

Your RTX 5070 has 12GB VRAM. For small models, the constraint isn't model size — it's batch size.

**Target effective batch size by scale:**

| Model Size | Effective Batch (tokens) | Micro-batch | Accumulation Steps |
|-----------|--------------------------|-------------|-------------------|
| 10K–100K | 32K–64K | 16K | 2–4 |
| 1M | 128K | 32K | 4 |
| 10M | 256K–512K | 32K–64K | 4–8 |
| 100M | 512K–1M | 64K–128K | 4–8 |

The general rule: batch size should scale roughly as sqrt(model_params) for optimal convergence. But at micro scale, even small batches work because gradient variance is lower with fewer parameters.

**Implementation tip:** With gradient accumulation, normalize the loss by accumulation steps, not by batch count:

```python
loss = model(batch) / accumulation_steps
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 2.4 Mixed Precision Training: FP32 vs BF16 vs FP8

**BF16 is your default.** The RTX 5070 (Blackwell, sm_120) supports FP32, BF16, FP16, FP8, and FP4 via 5th-gen Tensor Cores.

**Precision recommendations by component:**

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| Weights (master copy) | FP32 | Optimizer states need full precision |
| Forward/backward matmuls | BF16 | 2× speed, negligible quality loss |
| Attention softmax | FP32 accumulation | Numerical stability critical |
| RMSNorm | FP32 | Cheap, stability matters |
| Loss computation | FP32 | Precision matters here |
| Gradient accumulation | FP32 | Prevents underflow |

**FP8 Training on Blackwell:**

FP8 training is viable on your RTX 5070 via NVIDIA's Transformer Engine library. Key points:

- Transformer Engine (TE) supports FP8 on Hopper, Ada, and Blackwell architectures
- Uses delayed scaling: tracks activation statistics over recent steps to set per-tensor FP8 scale factors
- Two FP8 formats: E4M3 (for weights/activations in forward), E5M2 (for gradients in backward)
- Achieves near-BF16 convergence with measurably higher throughput

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.HYBRID)

# Replace nn.Linear with te.Linear
layer = te.Linear(in_features, out_features, bias=False)

with te.autocast(enabled=True, recipe=fp8_recipe):
    output = layer(input)
```

**Honest assessment for your use case:** At models under 100M params on a single GPU, FP8 training gives maybe 20–40% speedup in matmul-heavy layers. The engineering complexity (Transformer Engine integration, dealing with scaling factors, potential convergence issues at very small scale) may not be worth it until you're at the 100M param milestone. **Start with BF16. Move to FP8 at 100M if training time is your bottleneck.**

**FP4 (NVFP4) on Blackwell:** This is inference-only. Not usable for training.

### 2.5 torch.compile

**Use it. Seriously.** torch.compile (torch 2.x) with the `inductor` backend gives 10–30% training speedup with zero code changes to the model:

```python
model = torch.compile(model)  # That's it
```

**Benefits:** Kernel fusion (combine elementwise ops), reduced memory bandwidth pressure, automatic operator optimization.

**Pitfalls:**
- First iteration is slow (compilation). Use `mode="reduce-overhead"` for smaller models, `mode="max-autotune"` for 100M+ if you're training for hours
- Dynamic shapes (variable sequence lengths) cause recompilation. **Use fixed sequence lengths during training** (pack to a fixed context length)
- Some custom operations don't compile — you'll get a clear error
- On Blackwell GPUs, ensure you're using PyTorch nightly or 2.7+ with CUDA 12.8 (`pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`)

**PyTorch Blackwell support status (as of March 2026):** PyTorch nightly builds support RTX 5070 (sm_120) since Feb 2025. The stable 2.7+ releases work. Some third-party libraries (older Flash Attention versions, nerfstudio) had initial compatibility issues — always use latest versions.

### 2.6 Flash Attention

**FlashAttention-2 is your workhorse.** Pre-compiled wheels for Blackwell exist (flash_attn 2.7.4+, community builds on HuggingFace by marcorez8 and loscrossos). FA2 gives:
- O(N) memory instead of O(N²) for attention
- 2–3× speedup over naive attention at sequence length ≥512
- Critical for fitting longer contexts in 12GB VRAM

**FlashAttention-3:** Optimized for Hopper (H100) with warp-specialization and GEMM-softmax pipelining. As of March 2026, FA3 does NOT support Blackwell consumer GPUs (sm_120) — there's an open issue (#1853) on the Dao-AILab repo. FA3 is Hopper-only for now.

**FlashAttention-4:** Written in CuTe DSL, optimized for Blackwell B200 datacenter GPUs. Uses Tensor Memory and advanced warp specialization. Not available for consumer Blackwell (RTX 50-series) yet.

**For your pipeline: Use FlashAttention-2.** It works on Blackwell via community-compiled wheels. Alternatively, PyTorch's built-in `scaled_dot_product_attention` with `torch.compile` will automatically select an efficient backend (including Flash-like kernels) — this is the zero-dependency option.

```python
# Option A: PyTorch native (recommended for simplicity)
from torch.nn.functional import scaled_dot_product_attention
attn_output = scaled_dot_product_attention(q, k, v, is_causal=True)

# Option B: Flash Attention 2 explicit
from flash_attn import flash_attn_func
attn_output = flash_attn_func(q, k, v, causal=True)
```

---

## 3. Data Efficiency — Maximizing Learning per Token

### 3.1 Chinchilla Scaling Laws and Beyond

The Chinchilla optimal ratio is ~20 tokens per parameter for compute-optimal training. But this is for **compute-optimal** — if you care about final model quality and are willing to spend more compute (which you are, training on a single GPU), you should **overtrain**.

**Modern practice has moved far beyond Chinchilla:**

| Model | Params | Tokens | Tokens/Param | vs Chinchilla |
|-------|--------|--------|-------------|---------------|
| Chinchilla | 70B | 1.4T | 20:1 | 1× (baseline) |
| LLaMA 1 (7B) | 7B | 1T | 142:1 | 7× overtrained |
| LLaMA 3 (8B) | 8B | 15T | 1875:1 | 94× overtrained |
| Qwen3 (0.6B) | 0.6B | 36T | 60000:1 | 3000× overtrained |
| TinyLlama | 1.1B | 3T | 2727:1 | 136× overtrained |

The loss keeps improving past Chinchilla-optimal. For **inference-optimized** models (you want the best model you can deploy, not the cheapest to train), train much longer.

**For your pipeline:**

| Model Size | Chinchilla Optimal | Recommended (Overtrain) | Your Target |
|-----------|-------------------|------------------------|-------------|
| 10K | 200K tokens | 2M–10M tokens | 10M+ |
| 100K | 2M tokens | 20M–100M tokens | 100M+ |
| 1M | 20M tokens | 200M–1B tokens | 1B+ |
| 10M | 200M tokens | 2B–10B tokens | 5B+ |
| 100M | 2B tokens | 20B–100B tokens | 50B+ |

For a STEM-focused dataset, you'll run into data scarcity long before these targets. This is where data quality, repetition, and synthetic data come in.

### 3.2 Textbook-Quality Data: The Phi Approach

This is the single most impactful insight for your project. Microsoft's Phi series proved that **data quality can substitute for data quantity** at small scale:

- **Phi-1** (1.3B params): Trained on only 7B tokens (6B filtered web + 1B synthetic from GPT-3.5). Achieved 50.6% on HumanEval — competitive with models trained on 100× more data.
- **Phi-2** (2.7B params): Trained on ~1.4T tokens of "textbook-quality" data. Matched Mistral-7B and LLaMA-2 13B on reasoning benchmarks.
- **Phi-3** (3.8B params): Continued the approach with even more aggressive filtering. Matched GPT-3.5 quality.
- **Phi-4** (14B params): Trained on 9.8T tokens. Uses synthetic "textbook-like" data for math, coding, reasoning.

**The recipe:**
1. Filter web data aggressively for "educational value" — keep only content that reads like a textbook, lecture notes, or well-written explanation
2. Generate synthetic training data using a frontier LLM — have GPT-4/Claude generate textbook-quality explanations, worked problems, step-by-step derivations
3. Combine filtered web + synthetic data
4. The key insight: **a 10M param model trained on 100M tokens of textbook-quality STEM data will outperform the same model trained on 1B tokens of unfiltered web data**

**For your STEM pipeline specifically:**
- Physics: Synthetic derivations, worked problems with step-by-step solutions, concept explanations
- Chemistry: Balanced equations with explanations, mechanism descriptions, periodic table relationships
- Math: Proofs, problem-solution pairs, theorem statements with intuitive explanations
- Biology: Process descriptions, cause-effect chains, classification hierarchies

### 3.3 Curriculum Learning

**Mixed results, but worth trying for STEM.** The evidence:

- Bengio et al. (original curriculum learning paper) showed faster convergence when presenting easier examples first
- For language models, "easy" is hard to define — shorter sequences? Simpler vocabulary? Lower perplexity under a reference model?
- Some practitioners report modest improvements (~5% faster convergence) with no final quality difference
- For STEM specifically, a natural curriculum exists: definitions → formulas → derivations → complex problems

**Practical recommendation:** Start training on simpler STEM content (definitions, basic formulas, short factual statements) for the first 10–20% of training, then mix in full-complexity content. This is cheap to implement and unlikely to hurt.

### 3.4 Data Mixing Ratios for Multi-Domain STEM

No universally optimal ratio exists, but based on Phi and similar work:

| Domain | Suggested Mix | Rationale |
|--------|--------------|-----------|
| Mathematics | 30–35% | Reasoning-heavy, transfers broadly |
| Physics | 20–25% | Builds on math, quantitative reasoning |
| Chemistry | 15–20% | Domain-specific notation, less transferable |
| Biology | 15–20% | Mostly factual/descriptive content |
| General/Connective | 5–10% | Science writing, cross-domain explanations |

**Critical insight:** Math is over-weighted intentionally because mathematical reasoning capability transfers to all other domains. A model that's good at mathematical derivation will be better at physics, chemistry, and even biology reasoning.

### 3.5 Deduplication and Quality Filtering

**Dedup is mandatory.** Duplicate data causes memorization, wastes compute, and can create degenerate training dynamics.

Methods, from simplest to most effective:
1. **Exact dedup:** Hash each document, remove exact copies. Catches ~5–15% of web data.
2. **MinHash / LSH:** Near-duplicate detection. Catches paraphrased content. Use `datasketch` library.
3. **Substring dedup:** Remove documents that share long common substrings (>100 tokens). Catches copy-paste snippets.
4. **Quality filtering:** Score each document on "educational value" using a classifier trained on textbook vs. web data. The Phi approach.

For a curated STEM dataset that you're building yourself, exact dedup + manual quality review is sufficient. If pulling from web sources (FineWeb-Edu, The Pile, etc.), use MinHash dedup with a Jaccard threshold of 0.8.

### 3.6 Document Packing

**Best-fit packing beats simple concatenation.** When packing multiple documents into a fixed-length context window:

- **Padding:** Wastes tokens. Never do this.
- **Concatenation:** Join documents with separator tokens. Simple but leaks attention across document boundaries.
- **BOS-aligned best-fit packing** (nanochat approach): Pack documents into sequences aligned to BOS (beginning-of-sequence) tokens. Uses attention masking to prevent cross-document attention. More complex but cleaner.
- **Random concatenation with separator tokens:** The practical middle ground. Use `<|endoftext|>` between documents. At small context lengths (512–2048), cross-document attention leakage has minimal impact.

**Recommendation:** Use concatenation with separator tokens at ≤1M params. Switch to proper packing with attention masking at 10M+ where you're using longer contexts and care about quality.

### 3.7 Synthetic Data Generation for STEM

This is your secret weapon. Use Claude, GPT-4, or similar frontier models to generate:

1. **Worked problems:** "Solve this physics problem step-by-step: ..."
2. **Concept explanations:** "Explain [concept] as if writing a textbook section for undergraduates"
3. **Q&A pairs:** "Generate 10 questions and detailed answers about [topic]"
4. **Error correction:** "Here's an incorrect derivation. Identify and fix the errors."
5. **Multi-representation:** "Express this concept in words, then as an equation, then as a diagram description"

**Quality control for synthetic data:**
- Verify correctness of generated STEM content (especially math/physics — LLMs make calculation errors)
- Filter for diversity — don't let templates dominate
- Mix synthetic and real data (50/50 or 70% real / 30% synthetic is a common starting point)

---

## 4. Tokenization

### 4.1 Tokenizer Comparison by Model Scale

| Method | Pros | Cons | Best Scale |
|--------|------|------|------------|
| Character-level | Zero preprocessing, tiny vocab (~256), model must learn all structure | Very long sequences, slow training, limited context | <1M params (learning/debugging) |
| Byte-level (UTF-8) | No UNK tokens, handles any language | Even longer sequences than char-level | Research/experimentation only |
| BPE (standard) | Proven, efficient compression, widely used | Requires corpus to train tokenizer | 1M+ params |
| Unigram (SentencePiece) | Better morphological handling, principled probabilistic model | Slightly less common in LLM practice | 1M+ params |

**Honest assessment of character-level for your pipeline:** It's valuable as a learning exercise and for the micro/small milestones (10K–100K params). Character-level forces the model to learn everything from scratch — orthography, word boundaries, syntax — which is intellectually satisfying and educational. But it's **dramatically less efficient** than BPE. A character-level model needs ~4–5× more sequence length to represent the same text, which means:
- 4–5× more compute per document
- 4–5× less "conceptual content" per context window
- Much harder to learn long-range dependencies

**Switch to BPE at the 1M param milestone at the latest.** Your 100M param model should absolutely use BPE.

### 4.2 Optimal Vocabulary Sizes

| Model Scale | Recommended Vocab Size | Rationale |
|-------------|----------------------|-----------|
| <100K params | 256 (char-level) | Can't afford embedding matrix |
| 100K–1M | 256–1024 | Small BPE or char-level |
| 1M–10M | 4096–8192 | Good compression without oversized embeddings |
| 10M–100M | 8192–32768 | Balance between compression and embedding cost |
| 100M+ | 32768–65536 | nanochat uses 65536 at ~500M |

**The embedding tax:** With weight tying, the embedding matrix costs V × d_model parameters. For V=32768 and d_model=768, that's 25M parameters — 25% of a 100M param model. This is why smaller models need smaller vocabularies.

**Remember the vocab bottleneck:** Levine et al. proved that d_model should not significantly exceed vocab size, as the embedding matrix becomes rank-deficient. With a vocab of 256 (character-level), d_model of 256 is the practical maximum. This directly constrains your model width with character-level tokenization.

### 4.3 Implementation: tiktoken vs SentencePiece vs Custom

- **tiktoken** (OpenAI): Fast Rust-based BPE. Production-grade. Can train custom vocabularies using `tiktoken`'s training scripts.
- **SentencePiece** (Google): Supports both BPE and Unigram. More flexible. Good multilingual support.
- **nanochat's custom Rust BPE:** Minimal, fast, specifically designed for LLM training. 65K vocab, ~4.8 chars/token compression.
- **HuggingFace tokenizers:** Rust-backed, easy to train custom BPE/Unigram tokenizers. Good documentation.

**Recommendation:** Use HuggingFace `tokenizers` library to train a custom BPE tokenizer on your STEM corpus. This gives you control over special tokens and domain-specific merges.

### 4.4 Specialized Tokens for STEM

**Yes, add them.** Your tokenizer should include special tokens for:

```
<equation>...</equation>   — wrap mathematical equations
<formula>...</formula>      — chemical formulas
<proof>...</proof>         — mathematical proofs
<step>                     — step-by-step marker
<definition>               — definition marker
<theorem>                  — theorem statement
```

More importantly, when training the BPE tokenizer on STEM data, common mathematical expressions and chemical formulas will naturally emerge as tokens: `H2O`, `CO2`, `sin`, `cos`, `d/dx`, `∫`, `→`, `≈`, `∞`, etc.

**Train the tokenizer on your STEM corpus, not on general web data.** This ensures domain-relevant merges.

---

## 5. Evaluation and Understanding What the Model Learns

### 5.1 Metrics by Purpose

**Primary metric: Bits-per-byte (BPB).** Converts perplexity to a tokenizer-independent measure. Essential when comparing character-level vs BPE models:

```
BPB = (cross_entropy_loss × tokens) / bytes_in_text × log2(e)
```

This lets you fairly compare a character-level model against a BPE model on the same text. Lower is better.

**Perplexity:** Standard, but tokenizer-dependent. Only compare perplexity between models with the same tokenizer. For a STEM model, track perplexity separately per domain (math, physics, chemistry, biology).

**BLEU / ROUGE for generation:** Mostly useless for small models. These metrics need fluent, coherent output to be meaningful. Below 10M params, your model won't generate coherent text.

**Domain-specific probes:** The most informative evaluations for a STEM model:
- Can it complete common equations? (`E = mc` → `²`)
- Can it balance simple chemical equations?
- Can it state basic definitions? ("The mitochondria is the..." → "powerhouse of the cell")
- Can it do simple arithmetic in context?

### 5.2 Memorization vs Understanding

**The hardest question in small-model evaluation.** A 10M parameter model that outputs "F = ma" might be memorizing or might "understand" Newton's second law. Tests:

1. **Holdout perplexity:** Compare train vs test perplexity. Large gap = memorization. Small gap = generalization.
2. **Paraphrase test:** If the model knows "F = ma", can it complete "The force equals mass times..."?
3. **Novel combination test:** Present combinations of concepts not in training data. Can the model sensibly combine them?
4. **Probing classifiers:** Train small linear probes on intermediate representations to test what's encoded:
   - Does layer 3 encode part-of-speech?
   - Does layer 6 encode subject-domain (physics vs chemistry)?
   - Do attention heads show interpretable patterns?

```python
# Probing example
hidden_states = model.get_hidden_states(input_ids)  # (batch, seq, d_model)
probe = nn.Linear(d_model, num_classes)
# Train probe on frozen hidden_states to predict domain/concept
# High probe accuracy = information is encoded; low = not encoded
```

### 5.3 Scaling Laws: Predicting Large from Small

This is one of the most powerful tools for your pipeline. If you run multiple small models and track their loss:

**The parametric scaling law (Hoffmann et al.):**
```
L(N, D) = E + A/N^α + B/D^β
```
Where N = parameters, D = training tokens, E = irreducible loss, and α, β are exponents (~0.34, ~0.37).

**Practical workflow:**
1. Train models at 10K, 100K, 1M, 10M params to completion
2. Log final loss for each
3. Fit the scaling law parameters (E, A, B, α, β) to your data
4. **Predict** what loss a 100M param model should achieve
5. If your 100M model significantly underperforms the prediction, something is wrong (data, architecture, training)

This saves enormous time and compute — you catch problems at small scale before investing in large runs.

### 5.4 Evaluation at 9B+ Scale (Sovereign Pipeline)

At 9B+ parameters with post-training, evaluation shifts fundamentally from pretraining metrics to benchmark suites.

**From BPB to benchmarks.** During pretraining, bits-per-byte and perplexity are the primary signals -- they're cheap, fast, and correlate with downstream quality. After post-training (SFT, RL, distillation), these metrics become unreliable: a model can have higher perplexity but dramatically better reasoning. Benchmark accuracy on held-out problem sets becomes the ground truth.

**The Sovereign 10.** For the CogCore-9B-Sovereign pipeline, 10 primary benchmarks were selected for low contamination, remaining headroom, and coverage of target capabilities: GPQA Diamond (science), SuperGPQA (breadth), MMLU-Pro (knowledge+reasoning), AIME 2025 (math), LiveCodeBench v6 (code), BFCL-V4 (tool use), TAU2-Bench (agents), RULER (working memory), IFEval (instruction following), LongBench v2 (long context). Saturated benchmarks (GSM8K, MMLU original, HumanEval original) are excluded -- they no longer differentiate models at this scale.

**Quick eval vs Full eval.** A "quick eval" (~750 problems, <30 minutes) runs after every checkpoint to catch regressions. A "full eval" (~5-6 hours, all 10 benchmarks at full size) runs at phase boundaries. This tiered approach balances signal quality with iteration speed.

**Statistical rigor.** At 9B scale, claiming improvement requires more than eyeballing numbers. McNemar tests (paired, same problems) establish significance for "Model A beats Model B" claims. Wilson confidence intervals quantify uncertainty. Minimum sample sizes (500+ for model comparisons, 200+ for phase-over-phase) prevent false positives. If confidence intervals overlap, the claim is not supported.

**Contamination prevention.** With models trained on trillions of tokens, benchmark contamination is a real risk. Blocklists of exact benchmark texts, n-gram matching (8-gram), and SBERT similarity filtering (threshold 0.92) catch direct and paraphrased leakage. A paraphrase audit (re-test with modified problems) validates that improvements reflect genuine capability, not memorization.

**LLM-as-judge for custom domains.** Where no standard benchmark exists (first-principles physics, SAP/ABAP, financial analysis), a structured rubric + frontier LLM judge (calibrated against 20% human spot-check, Cohen's kappa > 0.7) provides scalable evaluation. This is essential for measuring the unique capabilities that differentiate a sovereign model.

See PLAN.md Section 7.5 for the full evaluation framework, schedule, and report card template.

---

## 6. Cutting-Edge Techniques (2025-2026)

### 6.1 nanochat (Karpathy)

nanochat (released Oct 2025, major update Jan 2026) is the single best reference implementation for your project. Key technical decisions:

- **Architecture:** GPT-2 style decoder-only transformer with modern improvements
- **Tokenizer:** Custom Rust BPE, 65536 vocab, ~4.8 chars/token
- **Training data:** FineWeb-Edu (filtered educational web data)
- **The `--depth` dial:** A single integer controls all hyperparameters. Depth 20 is a small model, depth 26 matches GPT-2 performance, depth 30+ gets into GPT-3 Small territory
- **Full pipeline:** Pretrain → Midtrain (chat, MCQ, tool use) → SFT → optional GRPO RL → inference with KV cache → web UI
- **Cost:** GPT-2 quality for $73 (3 hours on 8×H100), or $15 on spot instances
- **Latest (Jan 2026):** Midtraining stage removed, streamlined pipeline, GPT-2 grade in $73

**What to steal from nanochat:**
- The automatic hyperparameter scaling from depth
- The data pipeline (FineWeb-Edu shards, shuffled, streamed)
- The evaluation harness (CORE score, ARC-E/C, MMLU, GSM8K, HumanEval)
- The training script structure — clean, minimal, hackable

### 6.2 BitNet / 1-Bit LLMs

**BitNet b1.58 2B4T** (Microsoft, April 2025) is the landmark model: 2B parameters with ternary weights ({-1, 0, +1}), trained from scratch on 4T tokens. Key findings:

- Matches full-precision models of similar size on most benchmarks
- 0.4GB memory footprint vs 1.4–4.8GB for comparable models
- 29ms CPU decode latency (fastest in class)
- Uses BitLinear layers (replace nn.Linear), squared ReLU activation, RoPE, no biases
- Inference requires specialized kernels (bitnet.cpp) — standard PyTorch/transformers won't give speed benefits

**Does it work at small scale?** The BitNet paper shows results starting at 2B params trained on 4T tokens. There are no published results below 1B params.

**For your pipeline:** BitNet is primarily an **inference efficiency** technique. The training still uses full-precision master weights with ternary quantization applied during forward pass. On a single GPU, the training speed difference is minimal. The benefit comes at inference — a 100M param BitNet model would use ~25MB of weight storage and run blazingly fast on CPU.

**Recommendation:** Interesting experiment at the 100M milestone. Train a standard model AND a BitNet model, compare quality. But it's not your primary path — focus on standard architectures first.

### 6.3 Sparse Attention and KV-Cache Compression

**Sparse attention** (sliding window, dilated, BigBird-style) — useful for very long contexts but irrelevant at your scale. With 100M params and context length 512–2048, standard dense attention is fine and FlashAttention handles the memory.

**KV-cache compression** — matters only at inference with very long sequences. Techniques like GQA (which you should already be using), quantized KV-cache (FP8 keys/values), and Multi-Query Attention reduce inference memory. Implement GQA in your architecture; worry about further KV compression later.

**Speculative decoding** — uses a small "draft" model to propose tokens, validated by a larger model. Only relevant when you have two models of different sizes. Not applicable to your single-model pipeline, but interesting once you have your 10M and 100M models — the 10M model can draft for the 100M model.

### 6.4 Intelligence per FLOP: Recent Breakthroughs

The most impactful findings of 2025–2026 for small-scale training:

1. **Muon optimizer** (2× compute efficiency): Already covered in Section 2.1. The single biggest free lunch.

2. **Data quality over quantity** (Phi series): Already covered in Section 3.2. The second biggest free lunch.

3. **Overtraining small models** (LLaMA 3, Qwen3): Train 100–1000× beyond Chinchilla-optimal. The loss keeps going down. Qwen3-0.6B trained on 36T tokens (60,000:1 ratio) is the extreme example.

4. **Test-time compute** (o1/o3-style): Not directly about training efficiency, but the insight that you can get more "intelligence" out of a fixed model by spending more compute at inference (chain-of-thought, search, self-verification). Relevant for your STEM model — teach it to show its work.

5. **Precision Scaling Laws** (Raschka's 2024 highlight): Models trained on more data are harder to quantize. If you plan to deploy in low precision, don't overtrain as aggressively. There's a tradeoff between training tokens and post-training quantization quality.

---

## 7. Scale Milestone Recipes

### Milestone 1: 10K Parameters — "Hello World"

**Purpose:** Verify the pipeline works end-to-end.

```
Architecture: 2 layers, d_model=64, 2 heads, d_head=32, FFN=172 (GELU)
Tokenization: Character-level (vocab=256)
Positional encoding: Learned absolute (simple at this scale)
Normalization: Pre-RMSNorm
Weight tying: Yes
Parameters: ~10K (mostly in embeddings)

Data: 1M–10M characters of curated STEM text
Context length: 128 characters
Batch size: 32 sequences = 4096 tokens
Optimizer: AdamW (lr=1e-3, betas=(0.9, 0.95), wd=0.1)
Schedule: Cosine, 100 warmup steps
Precision: FP32 (model is tiny, no need for mixed precision)
Training time: ~5 minutes on RTX 5070

Expected result: Learns character frequencies, common character bigrams
Not expected: Coherent words or grammar
```

### Milestone 2: 100K Parameters — "Babbling"

```
Architecture: 4 layers, d_model=128, 4 heads, d_head=32, FFN=344 (GELU)
Tokenization: Character-level (vocab=256) or tiny BPE (vocab=1024)
Positional encoding: RoPE
Weight tying: Yes
Parameters: ~100K

Data: 10M–100M tokens
Context length: 256
Batch size: 64K tokens effective (gradient accumulation)
Optimizer: AdamW (lr=6e-4)
Schedule: Cosine, 500 warmup steps
Precision: BF16 mixed precision
Training time: ~30 minutes

Expected: Learns word boundaries, common words, basic grammar patterns
Not expected: Coherent sentences
```

### Milestone 3: 1M Parameters — "Grammar"

```
Architecture: 6 layers, d_model=256, 4 heads, d_head=64, SwiGLU FFN=688
Tokenization: BPE (vocab=4096)
Positional encoding: RoPE
Weight tying: Yes
Normalization: Pre-RMSNorm
Parameters: ~1M

Data: 100M–1B tokens of textbook-quality STEM
Context length: 512
Batch size: 128K tokens
Optimizer: Muon (matrices) + AdamW (rest)
Schedule: WSD (warmup 2%, stable 80%, decay 18%)
Precision: BF16
torch.compile: Yes
Training time: ~2–4 hours

Expected: Grammatical text, domain vocabulary, simple completions
Not expected: Factual accuracy, reasoning
```

### Milestone 4: 10M Parameters — "Knowledge"

```
Architecture: 12 layers, d_model=384, 6 heads (GQA: 6 query, 2 KV),
             d_head=64, SwiGLU FFN=1032
Tokenization: BPE (vocab=8192), trained on STEM corpus
Positional encoding: RoPE
Weight tying: Yes
Parameters: ~10M

Data: 1B–5B tokens (mix of real + synthetic STEM)
Context length: 1024
Batch size: 256K tokens
Optimizer: Muon (lr=0.02, wd=0.01) + AdamW (lr=3e-4, wd=0.1)
Schedule: WSD
Precision: BF16 + torch.compile
Flash Attention: Yes (via PyTorch SDPA or FA2)
Training time: ~12–24 hours

Expected: Completes STEM sentences coherently, knows common formulas,
          basic factual recall
Not expected: Multi-step reasoning, novel problem solving
```

### Milestone 5: 100M Parameters — "Reasoning"

```
Architecture: 16 layers, d_model=768, 12 heads (GQA: 12 query, 4 KV),
             d_head=64, SwiGLU FFN=2048
Tokenization: BPE (vocab=32768), trained on STEM corpus
Positional encoding: RoPE
Weight tying: Yes
Parameters: ~100M

Data: 10B–50B tokens (aggressively filtered + synthetic)
  — Math: 30%, Physics: 25%, Chemistry: 20%, Biology: 20%, Connective: 5%
Context length: 2048
Batch size: 512K–1M tokens
Optimizer: Muon (lr=0.02, wd=0.01) + AdamW (lr=3e-4, wd=0.1)
Schedule: WSD (warmup 1%, stable 85%, decay 14%)
Precision: BF16, consider FP8 via Transformer Engine
Flash Attention: FA2 or PyTorch SDPA
torch.compile: mode="max-autotune"
Training time: ~3–7 days on RTX 5070

VRAM budget (approximate):
  Model params (BF16): ~200MB
  Optimizer states (FP32 master + momentum): ~800MB
  Activations (with gradient checkpointing): ~2–4GB
  Batch data: ~1–2GB
  Total: ~5–7GB → fits comfortably in 12GB

Expected: Coherent STEM explanations, can complete equations,
          basic problem-solving in familiar patterns
Not expected: GPT-2 level fluency (that requires ~500M+ params)
```

---

## Appendix A: RTX 5070 Specs for Training

| Spec | Value | Implication |
|------|-------|-------------|
| Architecture | Blackwell (sm_120) | 5th-gen Tensor Cores |
| VRAM | 12GB GDDR7 | Fits up to ~500M params in BF16 with optimizer states |
| Memory bandwidth | ~672 GB/s | Decent for single-GPU training |
| FP32 TFLOPS | ~30 (estimated) | Baseline compute |
| BF16 Tensor TFLOPS | ~120 (estimated) | Primary training precision |
| FP8 Tensor TFLOPS | ~240 (estimated) | Available via Transformer Engine |
| TDP | 250W | Sustainable for multi-day training |
| PCIe | Gen 5 | CPU↔GPU transfer not a bottleneck |

**Key constraint:** 12GB VRAM limits batch size at 100M+ params. Use gradient checkpointing (`torch.utils.checkpoint`) to trade compute for memory when needed.

---

## Appendix B: Essential Code Patterns

### Training Loop Skeleton

```python
import torch
from torch.nn.functional import scaled_dot_product_attention

# Mixed precision setup
scaler = torch.amp.GradScaler('cuda')  # Only if using FP16; not needed for BF16

for step in range(total_steps):
    optimizer.zero_grad()
    
    for micro_step in range(gradient_accumulation_steps):
        batch = next(data_iter)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(batch) / gradient_accumulation_steps
        
        loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    if step % log_interval == 0:
        print(f"step {step}, loss {loss.item():.4f}, lr {scheduler.get_last_lr()[0]:.6f}")
```

### Muon Implementation (Simplified)

```python
import torch

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                       weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                
                if group['nesterov']:
                    update = grad + group['momentum'] * buf
                else:
                    update = buf
                
                # Newton-Schulz orthogonalization
                if update.dim() == 2:
                    update = newton_schulz_orthogonalize(update, group['ns_steps'])
                
                # Weight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update
                p.add_(update, alpha=-group['lr'])

def newton_schulz_orthogonalize(M, steps=5):
    """Approximate polar decomposition via Newton-Schulz iteration."""
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimized coefficients
    X = M / (M.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X
```

---

## Appendix C: Key References

1. **nanochat** — Karpathy (2025): github.com/karpathy/nanochat — Full-stack LLM training
2. **Muon Optimizer** — K. Jordan et al. (2024): github.com/KellerJordan/Muon
3. **Muon is Scalable for LLM Training** — Liu et al. (2025): arxiv.org/abs/2502.16982 — Moonlight model
4. **Chinchilla Scaling Laws** — Hoffmann et al. (2022): arxiv.org/abs/2203.15556
5. **Textbooks Are All You Need** — Gunasekar et al. (2023): arxiv.org/abs/2306.11644 — Phi-1
6. **Phi-3 Technical Report** — Abdin et al. (2024): arxiv.org/abs/2404.14219
7. **BitNet b1.58 2B4T** — Microsoft (2025): arxiv.org/abs/2504.12285
8. **FlashAttention-3** — Dao (2024): arxiv.org/abs/2407.08608
9. **FlashAttention-4** — modal.com/blog/reverse-engineer-flash-attention-4
10. **Which Transformer Architecture Fits My Data?** — Levine et al. (2021): arxiv.org/abs/2105.03928
11. **Transformer Engine** — NVIDIA: github.com/NVIDIA/TransformerEngine — FP8 training
12. **Turbo-Muon** — Boissin et al. (Dec 2025): Polar Express acceleration for Muon
13. **The Impact of Depth and Width** — OpenReview (2023): Depth vs width generalization
14. **MobileLLM** — Liu et al. (2024): Sub-billion parameter optimization for on-device
15. **Beyond Chinchilla-Optimal** — Sardana et al. (2023): Inference-aware scaling laws

---

*Report generated March 5, 2026. Based on publicly available research, community findings, and empirical results from the LLM training community.*
