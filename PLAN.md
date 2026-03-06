# CogCore-500M: The Plan
### Best IQ/parameter STEM reasoning model at 500M scale

---

## 1. Vision

A 500M model that reasons about the physical world from first principles. Not a fact database -- a reasoning engine that can re-derive F=ma from Newton's axioms, transfer mathematical structures across domains, and detect when answers violate conservation laws.

The model needs to internalize:
1. **Causal chains** -- why F=ma is true, not just the pattern
2. **Derivation capability** -- re-derive results from axioms
3. **Cross-domain transfer** -- same math governs fluid pressure, electrostatics, gravity
4. **Anomaly detection** -- spot conservation law violations before computing
5. **Multi-scale reasoning** -- quantum -> atomic -> chemistry -> biology

This comes from training data that explicitly demonstrates reasoning chains, not from textbook exposition.

---

## 2. Why 500M

Hardware: RTX 5070, 12GB GDDR7 VRAM, Blackwell sm_120.

| Config | VRAM | Batch | Training Time (80B tok) | Verdict |
|--------|------|-------|-------------------------|---------|
| 100M dense | 3 GB | 256 | 3 days | Too small |
| 300M dense | 4 GB | 128 | 8 days | Good baseline |
| **500M dense** | **5.5 GB** | **64** | **~16 days** | **Optimal** |
| 1B sparse MoE | 7.5 GB | 32 | 20 days | Max ceiling |
| 2B dense | 11.5 GB | 1-4 | 60+ days | Not worth it |

A 500M model trained on 80B tokens (160 tokens/param) will be competitive with 3-7B general-purpose models on STEM reasoning. Train smaller, train longer (Qwen3 lesson: 60K tokens/param).

Training memory (8-bit Muon, BF16, gradient checkpointing):
- Weights: 500M x 2B = 1.0 GB
- Gradients: 500M x 2B = 1.0 GB
- 8-bit Muon: 500M x 0.5B = 0.25 GB
- Activations (batch=32, seq=2048, checkpointed): ~1.5 GB
- CUDA overhead: ~1.5 GB
- Total: ~5.5 GB (6.5 GB headroom)

---

## 3. Final Architecture Spec

```
Model: CogCore-500M-v2 (Hybrid DiffTrans + Mamba2)

Core dims:
  d_model: 896
  n_layer: 32   (26 DiffTrans + 6 Mamba2)
  n_head: 14    (d_head=64)
  d_latent: 224 (MLA, d_model/4)

Attention (26/32 layers):
  Type: Differential Transformer (DiffTrans)
  KV compression: MLA latent bottleneck d=224
  QK-Norm: RMSNorm on Q and K before RoPE
  RoPE theta: 10,000 base + YaRN/NTK scaling for extension
  Logit softcap: 15 (CR2)
  Zero-init output projections (CR2)
  Long-short attention pattern (Addendum A):
    - Alternating layers: full-context and local-window attention
  Paired Head Attention (CR3-A1):
    - Heads paired, double key-space per pair, staggered RoPE
    - Apply to local-window layers only, start with 4 layers
  Sparse Attention Gate (CR3-A3):
    - Per-head sigmoid gate from first 12 dims of residual stream
    - Init gate weights to zero (gate fully open at start)
    - 0.1 multiplier for stability during early training
    - Exactly 12 input dims -- full d_model hurts vs no gate

SSM (6/32 layers, every 5th):
  Type: Mamba2
  State dim: d_model (896)

MLP (all 32 layers):
  Type: MoE SwiGLU
  n_experts: 8, top-2 routing
  expert_hidden: 448 (d_model/2)
  shared_expert_hidden: 224

Skip connections:
  SkipV1: layer-1 V reused via learnable alpha (init=0.0)
  U-net skip connections (Addendum A)

Embeddings:
  Value embeddings (Addendum A)
  Bigram hash embeddings (Addendum A)
  Smear token 1-position (Addendum A)

Normalization:
  Pre-RMSNorm on residual stream
  QK-Norm on Q and K
  nGPT unit norms on embedding + output weight matrices

Other:
  No weight tying, no biases
  BPE 8192, context 2048

Total params: ~490M sparse, ~350M active at inference
```

### Backout Mechanism (CR3-A2)

Before the final lm_head prediction, subtract a learned fraction of the residual stream captured at 2/3 depth. Lets the model clean up noisy early-layer signals that helped build context but hurt the final prediction.

- `backout_lambda`: learned scalar, init=0 (no-op at start)
- `backout_layer_fraction`: 0.667 (capture at 2/3 depth)
- Stop gradient through backout path (`x_backout.detach()`)

### Partitioned Hyperconnections (CR3-A4) -- IMPLEMENT LAST

Split the residual stream into two parallel lanes from layer 7 onward. Attention reads lane0, MLP reads lane1. Both write back to both lanes via learned weights. Early layers (0-6) stay single-lane.

- Per-layer learned write weights (w_post0..3) control cross-lane mixing
- Per-layer residual decay with exponential init by depth
- Final output: average of both lanes
- **IMPLEMENT LAST** among all architecture changes -- most complex, most recent. Verify all other changes are stable first.

### Why these choices

- **DiffTrans over standard MHA**: +7.5% avg reasoning, needs only 65% params to match baseline (ICLR 2025)
- **Mamba2 hybrid (6/32 layers)**: +2% ARC, 2-8x faster inference on long contexts
- **32L x d=896 over 24L x d=1024**: deeper = more sequential reasoning steps, same param count
- **QK-Norm**: adopted by OLMo2, Gemma3, every serious 2025 small model. Zero overhead.
- **nGPT unit norms**: 4x faster convergence, reduces warmup needed
- **SkipV1**: 25-50% KV-cache reduction + better perplexity via first-layer V reuse
- **Paired Head Attention**: richer position mixing in one softmax, zero extra params (-65 steps in nanogpt speedrun)
- **Backout**: decouples context-building from prediction, suppresses early-layer noise
- **Sparse Attention Gate**: prevents attention sink, heads learn to no-op when not useful
- **Hyperconnections**: two-lane residual gives attention and MLP independent read paths

---

## 4. Data Strategy

**Split: 32.5% real / 67.5% synthetic, 80B tokens total**

### Real sources (32.5% = 26B tokens)
- Proof-Pile-2 (math + CS papers)
- AlgebraicStack (verified math code)
- MathCode-Pile (math + code)
- OpenWebMath (web math content)
- Big-Math (>250k verified problems, replaces raw NuminaMath-CoT)
- MathPile (curated math corpus)
- AutoMathText V2 (`OpenSQZ/AutoMathText-V2`) -- quality-filtered math+STEM, 2.46T tokens available. LLM-scored quality (not just perplexity/keyword). Content: math web, arXiv, GitHub math code, StackExchange, NuminaMath subsets, MetaMath QA. (CR3-C2)

### DeepSeekMath-style CC Mining (CR3-C1)

Iterative fastText classifier to mine high-quality STEM content from Common Crawl, producing 5-9x more math tokens from the same source as OpenWebMath.

Pipeline:
1. Round 1: positives=OpenWebMath samples, negatives=random CC pages, train fastText, keep top 5% confidence
2. Round 2: human review top-yield URL patterns/domains, add confirmed domains as positive seeds, retrain, keep top 5%
3. Round 3: refine with hard negatives, MD5 dedup + benchmark decontamination

Target: expand real data from ~26B to ~50B tokens before ClimbMix clustering.

### Synthetic types (67.5% = 54B tokens)

| Type | % of total | Tokens |
|------|-----------|--------|
| Math+code hybrids (derivation + Python verification) | 10% | 8B |
| Physics/chemistry simulation code | 5% | 4B |
| First-principles derivations | 12.5% | 10B |
| Cross-domain bridges | 7.5% | 6B |
| Formal proof + intuition pairs | 7.5% | 6B |
| Process-supervised step traces | 5% | 4B |
| Competition math solutions | 7.5% | 6B |
| Socratic chains | 5% | 4B |
| Rephrased versions | 2.5% | 2B |
| Paradox resolutions | 5% | 4B |
| **SYNTHETIC TOTAL** | **67.5%** | **54B** |

### PRM-Verified Synthetic Data Filter (CR3-C3)

Before adding ANY synthetic training example, verify EVERY reasoning step using a Process Reward Model (not just final answer). Keep example only if all steps score above threshold (e.g., 0.7).

PRM options: math-shepherd-mistral-7b-prm (open source), Qwen-Math-PRM variants, or Claude API as lightweight proxy for seed examples.

Why: outcome-only filtering misses correct-answer-wrong-reasoning examples. PRM filtering eliminates this. +2pp on math reasoning benchmarks.

Apply to: all 10 synthetic data types, especially first-principles derivations and competition math solutions. Not needed for real data.

### Complete Problem-Solution Pairs (CR3-C4)

NEVER split a problem-solution pair across sequence boundaries. Each training sequence must contain complete reasoning trajectories. Non-reasoning docs can split normally.

Cross-sequence reasoning chains teach the model to reason across document boundaries -- pure noise. Complete trajectories in single sequences teach coherent chains. One of the highest-impact data hygiene changes.

### Pipeline
1. **ClimbMix**: embed all real data with small encoder, K-means into 20-25 clusters, train 50M proxy to find optimal cluster weights
2. **MATES**: BERT-base proxy computes influence scores every 500 steps during pretraining, dynamically upweights high-influence data
3. **RHO-1**: selective token weighting from step 0 (not fine-tuning only). Train reference model, upweight tokens where excess loss is high.

### What's NOT in pretraining
- Lean 4 formal proofs: no measurable cross-domain transfer to informal STEM at <1B scale. Post-training only if at all.

---

## 5. Training Config

```python
# Optimizer: Cautious 8-bit Muon with Polar Express + MuonClip
optimizer = Cautious_8bit_Muon(
    matrix_params=[...],  # all 2D weights except embed/norm
    lr=0.02,
    momentum=0.95,
    weight_decay=0.1,       # base value, scaled by LR ratio (CR3-B2)
    nesterov=True,
    ns_steps=5,
    quantize_8bit=True,
    cautious=True  # mask updates where grad/momentum disagree
)
# Embed, norms: standard AdamW

# QK-Clip (MuonClip)
qk_clip_max_norm = 10.0

# Schedule: WSD
warmup_steps = 500      # reduced from 1000 (nGPT faster convergence)
stable_steps = 41000
decay_steps = 9000
min_lr = 3e-5

# Seq length schedule (CR2): 256 -> 512 -> 1024 -> 2048
# Batch size schedule (CR2): 64 -> 128 -> 256

# Training objectives (all active simultaneously):
loss = (
    ntp_loss                          # standard next-token prediction
    + 0.3 * mtp_loss                  # MTP-4 sequential heads
    + 0.1 * fsp_loss                  # Future Summary Prediction, window=64
    + rho1_selective_weighting        # excess loss token weighting
    + 0.05 * process_supervision_loss # step-level correctness signal
)

# Scale
batch_size = 32, gradient_accumulation = 8  # effective batch = 256
seq_len = 2048
tokens/step = 524,288
total = 80B tokens -> ~152,000 steps -> ~16 days on RTX 5070
```

### Polar Express (CR3-B1)

Drop-in replacement for Newton-Schulz 5 in Muon's orthogonalization step. Better polynomial approximation for polar decomposition in bfloat16. Lower final loss, same compute cost.

Action required: pull exact coefficients from modded-nanogpt PR #172 / record 38 commit. Placeholder coefficients in CR3 are (3.4445, -4.7750, 2.0315) -- update with actual refined values.

### Cautious Weight Decay with LR-Tied Schedule (CR3-B2)

Weight decay scales with current learning rate ratio, not fixed. During WSD decay phase, weight decay drops proportionally:

```
wd_scale = current_lr / peak_lr   # 1.0 at peak, ~0.1 at end
effective_wd = base_wd * wd_scale  # 0.1 * ratio
```

Apply to both Muon and Adam optimizers. Prevents over-regularization during final fine-grained adjustments.

### Embedding-Specific Gradient Accumulation (CR3-A5)

Embedding matrix and lm_head accumulate gradients for 2 optimizer steps before updating. Rest of network updates every step. Embedding gradients are high-variance (each token appears infrequently); accumulating gives more stable gradient estimates.

### Progressive Embed/lm_head Untying (CR2)

Gradual untying of embedding and lm_head weights during training.

---

## 6. Post-Training

### Stage 1: Evaluate Base Model (CR3-D1)

**Mandatory pre-RL quality threshold.** Before starting ANY RL, evaluate the base pretrained model:

```
GSM8K zero-shot:  > 20%   (grade school math)
MATH-500:         > 10%   (competition math)
ARC-Challenge:    > 40%   (science reasoning)

If any metric is BELOW these thresholds:
  -> Do NOT start GRPO/Dr. GRPO
  -> Return to pretraining: add more math/reasoning data, train more
  -> Re-evaluate after 10B more tokens

These are MINIMUMS for RL to be useful.
Below these: RL produces length bloat, not reasoning.
```

### Stage 2: Cold-Start CoT (MANDATORY before any RL)

Cold-start SFT is NOT optional at 500M -- it is mandatory. R1-Zero style (pure RL from scratch) does NOT work at 500M. It produces length bloat and instability. The "aha moment" / spontaneous reasoning emergence requires ~1.5B+ params.

- Generate 5,000 Chain-of-Thought traces
- **Magpie-style self-synthesis (CR3-C5)**: prompt Claude with STEM system prompt + empty user turn to generate both question AND answer. More diverse than template-based generation.
- Mix: 60% Magpie-style (diverse), 40% targeted hard problems (AIME/IMO level)
- **Include false starts and self-corrections (CR3-D2)**: 60% traces with self-correction, 40% clean correct traces
- Format: `<think>...reasoning with false starts...</think><answer>...</answer>`
- SFT fine-tune -> CogCore-500M-Thinking

Why false starts: models trained on clean-only traces learn straight-line derivations. Models trained with self-correction learn error detection and recovery -- exactly what RL later reinforces.

### Stage 3: Dr. GRPO Reasoning RL

**Dr. GRPO configuration -- the empirically validated subset for <1B models.**

Full DAPO hurts at 500M (Qwen2.5-0.5B case study: Clip-Higher and dynamic sampling specifically degrade performance at <1B scale).

```
KEEP (Dr. GRPO):
  - Token-level policy gradient loss (not sample-level)
  - Overlong filtering (mask reward on truncated responses)
  - No KL divergence penalty (rely on clipping)
  - Symmetric clipping: eps = 0.2

REMOVED (hurts at 500M):
  - Asymmetric clipping (eps_high=0.28) -- reverted to symmetric
  - Dynamic sampling / zero-variance masking -- degrades at this scale
```

- **Verifiable Reward Gym (CR3-D3)**: domain-specific reward functions
  - Mathematics: exact match via sympy
  - Physics numerical: relative tolerance +/-1%
  - Physics symbolic: sympy symbolic equivalence
  - Chemistry equations: atom + charge balance verification
  - Proofs: LLM judge for required logical steps
  - Non-verifiable problems: exclude from RL entirely (use in SFT only)
- E2H curriculum: easy-to-hard problem scheduling
- 3,000-5,000 RL steps -> CogCore-500M-Reasoner

### Stage 4: SLERP Merge (CR3-D4, MANDATORY)

After RL, merge the RL checkpoint back into the pretrained base using SLERP. This is NOT optional.

Small models (500M) lose 10-30% of pretrained knowledge during RL. Merging recovers most lost knowledge while keeping ~80-95% of reasoning gains.

```
SLERP(CogCore-500M-Base, CogCore-500M-Reasoner, alpha=0.6)
-> CogCore-500M-Final

alpha=0.6: 60% RL model, 40% base (empirically validated sweet spot for 500M)
```

### Stage 5: On-Policy Self-Distillation (optional)

- Student generates 8 responses per problem
- Score with rule-based verifier or teacher model
- SFT on top-k responses, repeat 3-5 rounds
- On-policy dominates off-policy by 4-8x token efficiency
- -> CogCore-500M-Distilled

### Inference: Best-of-N Sampling (CR3-D5)

Best-of-32 sampling with verifier at inference time. Highest-ROI test-time compute method for 500M.

- N=32: biggest relative gains for 500M (scaling curve steeper than large models, plateaus ~64-128)
- Temperature=0.8, max_tokens=2048, 1000+ token extended thinking
- Use Verifiable Reward Gym functions as verifier
- Fallback: model self-scoring for open-ended problems
- Alone gives 15-25% relative improvement on MATH/AIME vs greedy decoding

---

## 7. Phase Checklist

### Phase 1: BPE Tokenizer -- DONE
- [x] HuggingFace `tokenizers` library, BPE vocab=8192
- [x] `individual_digits=True`, byte-level fallback
- [x] Train on STEM corpus
- [x] Integrate into codebase, backward-compatible with char tokenizer
- [x] 5.0x compression ratio achieved

### Phase 2: Data Pipeline
- [ ] DeepSeekMath-style CC mining with fastText classifier (CR3-C1)
- [ ] AutoMathText V2 download and quality-score sampling (CR3-C2)
- [ ] ClimbMix semantic clustering (20-25 clusters)
- [ ] Synthetic data generation (10 types)
- [ ] PRM verification of all synthetic examples (CR3-C3)
- [ ] Document packer with complete problem-solution pairs (CR3-C4)
- [ ] Quality filtering + dedup

### Phase 3: Architecture (implement in order)
1. QK-Norm
2. DiffTrans attention + MLA
3. 32L x d=896 dimensions
4. Mamba2 hybrid layers (6/32)
5. Long-short attention pattern (Addendum A)
6. Logit softcap=15 (CR2)
7. Zero-init output projections (CR2)
8. Value embeddings (Addendum A)
9. U-net skip connections (Addendum A)
10. Bigram hash embeddings (Addendum A)
11. Smear token 1-position (Addendum A)
12. Sparse attention gate (CR3-A3)
13. Backout mechanism (CR3-A2)
14. Paired head attention (CR3-A1)
15. Partitioned hyperconnections (CR3-A4) -- LAST

### Phase 4: Training Infrastructure
- [ ] Cautious 8-bit Muon optimizer with Polar Express (CR3-B1)
- [ ] Cautious weight decay with LR schedule (CR3-B2)
- [ ] Embedding-specific gradient accumulation (CR3-A5)
- [ ] Progressive embed/lm_head untying (CR2)
- [ ] WSD schedule (warmup=500)
- [ ] Seq length schedule: 256->512->1024->2048 (CR2)
- [ ] Batch size schedule: 64->128->256 (CR2)
- [ ] MTP-4 + FSP-64 heads
- [ ] RHO-1 selective weighting from step 0
- [ ] Process supervision loss

### Phase 5: Pretraining Run
- [ ] 80B tokens, ~16 days on RTX 5070
- [ ] MATES dynamic data selection
- [ ] EoS document alignment (CR2)
- [ ] Monitoring: loss, gradients, attention entropy
- [ ] -> CogCore-500M-Base

### Phase 6: Post-Training
- [ ] Evaluate base model: GSM8K>20%, MATH>10%, ARC>40% (CR3-D1)
- [ ] Cold-start CoT with Magpie-style generation (CR3-C5/D2)
- [ ] Dr. GRPO with Verifiable Reward Gym (CR3-D3)
- [ ] SLERP merge (CR3-D4, mandatory)
- [ ] On-policy self-distillation (optional)
- [ ] Best-of-32 inference setup (CR3-D5)

---

## 8. Technique Table

| Technique | Source | Phase | Expected Gain |
|-----------|--------|-------|--------------|
| BPE 8192 (STEM-tuned) | All 2026 models | 1 | 3-5x compression |
| DeepSeekMath CC mining | arXiv:2402.03300 | 2 | 5-9x more math tokens |
| AutoMathText V2 | arXiv:2402.07625 | 2 | Quality-filtered 2.46T pool |
| ClimbMix clustering | NVIDIA 2025 | 2 | 2-3x data efficiency |
| PRM-verified synthetic filter | MathShepherd 2025 | 2 | +2pp math reasoning |
| Complete sequence packing | Multiple 2025-2026 | 2 | Data hygiene, coherent chains |
| MATES dynamic selection | NeurIPS 2024 | 5 | +1.1-1.3% zero-shot |
| RHO-1 pretraining | RHO-1 paper | 4 | +6.8% avg 15 tasks |
| Process-supervised traces | Rationalyst 2025 | 2 | +3.9% generalization |
| DiffTrans attention | ICLR 2025 | 3 | +7.5% reasoning |
| Mamba2 hybrid | Nemotron-H | 3 | +2% ARC, 2-8x inference |
| 32L x d=896 | SmolLM3/IMU-1 | 3 | +1-3% hard reasoning |
| QK-Norm | OLMo2/Gemma3 | 3 | Stability, free |
| nGPT unit norms | arXiv 2410.01131 | 3 | 4x faster convergence |
| SkipV1 connections | NeurIPS 2025 | 3 | 25-50% KV-cache reduction |
| Long-short attention | Addendum A | 3 | Efficient local+global mix |
| Logit softcap=15 | CR2 | 3 | Stability |
| Zero-init output proj | CR2 | 3 | Clean init |
| Value embeddings | Addendum A | 3 | Richer token repr |
| U-net skip connections | Addendum A | 3 | Better gradient flow |
| Bigram hash embeddings | Addendum A | 3 | Subword context |
| Smear token 1-position | Addendum A | 3 | Position smoothing |
| Paired head attention | modded-nanogpt rec 58 | 3 | Richer position mixing |
| Backout mechanism | modded-nanogpt rec ~40 | 3 | Cleaner predictions |
| Sparse attention gate | modded-nanogpt rec 28 | 3 | Prevents attention sink |
| Partitioned hyperconnections | modded-nanogpt rec 73 | 3 | Independent read paths |
| Cautious 8-bit Muon | arXiv 2411.16085 | 4 | Measurable perplexity |
| Polar Express | modded-nanogpt PR #172 | 4 | Better polar decomp in bf16 |
| Cautious WD schedule | modded-nanogpt rec 43+50 | 4 | Less over-regularization |
| Embed gradient accum | modded-nanogpt | 4 | Stable embed updates |
| Progressive embed untying | CR2 | 4 | Gradual capacity split |
| WSD schedule | DeepSeek V3 | 4 | Best schedule |
| Seq/batch size schedules | CR2 | 4 | Curriculum-style scaling |
| MTP-4 (0.3) + FSP-64 (0.1) | DeepSeek/Meta | 4 | Multi-objective |
| Pre-RL quality threshold | Research report Q5 | 6 | Gate bad RL starts |
| Cold-start CoT (mandatory) | Multiple 2025-2026 | 6 | Required for 500M RL |
| Magpie-style self-synthesis | ICLR 2025 | 6 | Diverse CoT generation |
| False starts in CoT | Research report Q8 | 6 | Error recovery learning |
| Dr. GRPO (stripped DAPO) | ByteDance/Qwen2.5 study | 6 | Same result, fewer steps |
| Verifiable Reward Gym | Kimi K2 methodology | 6 | Multi-domain RL rewards |
| E2H curriculum | arXiv 2506.06632 | 6 | Generalization boost |
| SLERP merge (mandatory) | Research report Q13 | 6 | Recover lost knowledge |
| On-policy distillation | 2025-2026 trend | 6 | 4-8x token efficiency |
| Best-of-32 inference | Research report Q12 | 6 | 15-25% relative improvement |

## 9. Expected Outcomes

| Metric | 1M (done) | After BPE + data | CogCore-500M | After Dr. GRPO |
|--------|-----------|------------------|--------------|----------------|
| Loss | ~2.5 (chars) | ~1.8 (BPE) | ~1.0 | <1.0 |
| Coherent STEM answers | No | Partial | Yes | Yes, verified |
| Multi-step reasoning | No | No | Yes | Yes, improved |
| Follows instructions | No | No | Partial | Yes |
| Domain transfer | None | Weak | Moderate | Strong |

**Net result:** A 500M model that performs like a 1.5-2B standard transformer on STEM reasoning. The IQ-first, facts-from-tools framing is the correct goal.

## What NOT to Do at This Scale

| Technique | Why Skip |
|-----------|----------|
| Full DAPO (asymmetric clip, dynamic sampling) | Hurts at 500M (Qwen2.5-0.5B case study) |
| R1-Zero style (pure RL, no cold-start) | Length bloat, no "aha moment" below 1.5B |
| RLHF with human raters | Cost of reward model > value at this scale |
| Multi-agent reasoning | Context too small, model too weak to self-critique |
| RAG | Retrieval overhead >> model itself |
| FP8 training | <5% gain, high complexity |
| Lean 4 in pretraining | No cross-domain transfer at <1B |
| Constitutional AI | Confirmed non-viable at 500M -- permanently dropped |

## What Remains for Change Request 4

Pending further research or implementation experience:
- Exact Polar Express polynomial coefficients (need PR #172 exact values)
- Self-play data bootstrapping (PretrainZero / LSP -- still emerging)
- FAC-SAE diversity metrics for synthetic data validation
- Multi-turn reasoning RL (emerging, not yet mature at 500M)

---

*All changes backed by 2025-2026 empirical results at <1B scale.*
*Primary sources: DiffTrans (ICLR 2025), Nemotron-H (2504.03624), ClimbMix (NVIDIA 2025), MATES (NeurIPS 2024), DAPO (ByteDance 2503.14476), Cautious-Adam (arXiv 2411.16085), 8-bit Muon (2509.23106), E2H Reasoner (2506.06632), Rationalyst (2025), SkipV1Former (NeurIPS 2025), nGPT (arXiv 2410.01131), modded-nanogpt PRs #117/#130/#140/#172/#191/#230, DeepSeekMath arXiv:2402.03300, AutoDS ACL 2025, OPSD arXiv:2601.18734, STILL-3 RUCAIBox 2025, MathShepherd 2025, Qwen3 technical report 2025.*
