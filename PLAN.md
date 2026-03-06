# CogCore-1B: The Plan
### Best IQ/parameter STEM reasoning model at 1B scale

---

## 1. Vision

A 1B model that reasons about the physical world from first principles. Not a fact database -- a reasoning engine that can re-derive F=ma from Newton's axioms, transfer mathematical structures across domains, and detect when answers violate conservation laws.

The model needs to internalize:
1. **Causal chains** -- why F=ma is true, not just the pattern
2. **Derivation capability** -- re-derive results from axioms
3. **Cross-domain transfer** -- same math governs fluid pressure, electrostatics, gravity
4. **Anomaly detection** -- spot conservation law violations before computing
5. **Multi-scale reasoning** -- quantum -> atomic -> chemistry -> biology

This comes from training data that explicitly demonstrates reasoning chains, not from textbook exposition.

---

## 2. Why 1B

Hardware: RTX 5070, 12GB GDDR7 VRAM, Blackwell sm_120.

| Config | VRAM | Batch | Training Time (80B tok) | Verdict |
|--------|------|-------|-------------------------|---------|
| 100M dense | 3 GB | 256 | 3 days | Too small |
| 300M dense | 4 GB | 128 | 8 days | Good baseline |
| **1B dense interleaved hybrid** | **~8-10 GB** | **8** | **~50-70 days** | **Optimal** |
| 2B dense | 11.5 GB | 1-4 | 60+ days | Not worth it |

A 1B interleaved hybrid dense model with Mamba-3 + periodic GQA. Falcon-H1 proves hybrid architecture class beats pure transformers 7-14x at 0.5B. With Mamba-3 + periodic GQA at 1B, target 7-10B dense equivalent on STEM, ceiling 12B on targeted benchmarks with test-time compute. Train smaller, train longer (Qwen3 lesson: 60K tokens/param).

Training memory (8-bit NorMuon + AdamW, BF16, FA2, gradient checkpointing):
```
Weights: ~1B x 2B = 2.0 GB
Gradients: ~1B x 2B = 2.0 GB
Optimizer (8-bit NorMuon + AdamW): ~2.0 GB
Activations (FA2 + grad checkpoint, seq=4096): ~0.5 GB
SSM states (32 layers): ~0.5 GB
CUDA overhead: ~1.5 GB
Total: ~8.5 GB (3.5 GB headroom on 12GB RTX 5070)
```

Notes: FA2 only (FA3 requires Hopper sm_90, RTX 5070 is Blackwell sm_120). FP8 for matmuls, BF16 for SSM states.

---

## 3. Final Architecture Spec

```
CogCore-1B v5 -- Interleaved Hybrid Dense (Mamba-3 + periodic GQA)

Core dims:
  d_model:       1280
  n_layer:       32 (26 SSMBlock + 6 HybridBlock)
  n_head:        20 query / 4 KV (GQA, 5:1 ratio) [hybrid blocks only]
  d_head:        64
  hidden_mlp:    3392 (d_model x 8/3, rounded to 64)
  vocab_size:    128000
  embed_dim:     640 (projected to d_model via Linear(640, 1280))
  context:       4096

Block layout (32 layers):
  [SSM, SSM, SSM, SSM, HYBRID(full), SSM, SSM, SSM, SSM, HYBRID(local),
   SSM, SSM, SSM, SSM, HYBRID(full), SSM, SSM, SSM, SSM, HYBRID(local),
   SSM, SSM, SSM, SSM, HYBRID(full), SSM, SSM, SSM, SSM, HYBRID(local),
   SSM, SSM]
  Hybrid at indices: 4, 9, 14, 19, 24, 29

SSMBlock (26 layers):
  - Pre-RMSNorm -> Mamba-3 MIMO (state=1280, d_conv=4, expand=2)
  - Pre-RMSNorm -> Dense SwiGLU MLP (hidden=3392, zero-init down_proj)

HybridBlock (6 layers):
  - Pre-RMSNorm -> PARALLEL:
      Path A: Mamba-3 MIMO
      Path B: GQA (20Q/4KV, QK-Norm, RoPE, softcap=15, partial key offset=1)
  - Gated addition: gate_ssm * A + gate_attn * B (gates init 0.5/0.5)
  - Pre-RMSNorm -> Dense SwiGLU MLP

Attention extras (hybrid blocks only):
  Long-short: full attn at layers 4,14,24; local window=512 at 9,19,29
  Sparse gate: per-head sigmoid gate, local-window layers only (CR3-A3)
  Paired heads: on local-window layers (CR3-A1)
  Value embeds: (128000, 64) with learned gate, injected into V path

Embeddings:
  Token:       (128000, 640) -> Linear(640, 1280) projection
  Bigram hash: (128000, 640) -> Linear(640, 1280) projection
  Smear:       gated 1-position look-back

Skip connections:
  U-net: gated (init 0)
  SkipV1: layer-1 V reuse (hybrid blocks only)
  Backout: residual capture at layer 21 (2/3 depth)

Normalization:
  Pre-RMSNorm, QK-Norm on Q/K, nGPT unit norms

Initialization:
  Output projections: zeros
  Embed/lm_head: tied at start, untied at 2/3 training
  SSM: default Mamba-3 init
  Merge gates: 0.5/0.5
  Value embed gates: 0
  Skip gates: 0

Param budget:
  Token embed + proj:    ~83M
  Bigram hash + proj:    ~83M
  26 SSMBlocks:          ~676M
  6 HybridBlocks:        ~182M
  Norms/gates/misc:      ~5M
  TOTAL:                 ~1.03B (all active, 100% utilization)
```

### Backout Mechanism (CR3-A2)

Before the final lm_head prediction, subtract a learned fraction of the residual stream captured at 2/3 depth (layer 21). Lets the model clean up noisy early-layer signals that helped build context but hurt the final prediction.

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

- **Interleaved hybrid over all-parallel**: 5:1 SSM:attention ratio validated in Mamba-3 paper at 1.5B. Saves ~40% VRAM vs 32 full-attention layers. Mamba-3 handles local/sequential; periodic GQA handles global.
- **GQA over DiffTrans**: Zero frontier adoption of DiffTrans. GQA used by every major lab (Qwen, DeepSeek, Meta, NVIDIA). Battle-tested at 0.8B-1.5B.
- **GQA over MLA**: MLA not validated below 7B. Extra latent projections waste params at 1B/4096 context. GQA simpler, proven, lower overhead.
- **Gated addition over MergeAttention**: Hymba-proven merge strategy. No cross-attention overhead. Learned gates allow model to balance SSM vs attention per layer.
- **Mamba-3 MIMO over Gated DeltaNet**: +0.6-1.8pp accuracy at 1.5B (Mamba-3 paper). Superior state-tracking and retrieval.
- **128K vocab over 32K**: 2026 frontier minimum is 65K. 128K gives ~15-20x byte compression. 2-3x more content per context window. Strong trilingual coverage (Serbian Cyrillic, German umlauts, English). Embedding projection prevents param bloat.
- **WSD over cosine**: Stable LR phase enables checkpoint flexibility. 2026 frontier consensus (Kimi K2, DeepSeek).
- **CoT distillation**: THE technique for 1B reasoning. MobileLLM-R1 proves 950M can hit MATH 74% through teacher distillation. Without it, 1B models cap at ~40-50% MATH.
- **4096 context over 2048**: Feasible on 12GB (confirmed). Doubles reasoning chain length. Critical for CoT emergence.
- **Dense over MoE**: 100% param utilization. March 2026 consensus -- every major lab uses dense at sub-1B.
- **nGPT unit norms**: 4x faster convergence, reduces warmup needed
- **SkipV1**: 25-50% KV-cache reduction + better perplexity via first-layer V reuse
- **Paired Head Attention**: richer position mixing in one softmax, zero extra params
- **Backout**: decouples context-building from prediction, suppresses early-layer noise
- **Sparse Attention Gate**: prevents attention sink, heads learn to no-op when not useful
- **Hyperconnections**: two-lane residual gives attention and MLP independent read paths

---

## 4. Data Strategy

**Primary source: FineWeb-Edu** (publicly available, proven). ClimbMix (NVIDIA Nemotron) is SOTA but not publicly available as of March 2026 -- switch if it becomes public during training.

**Domain ratios (reasoning-optimized):** 30% code, 25% math/science, 25% general web (filtered), 10% multilingual (Serbian/German), 10% synthetic reasoning traces.

**Total: 80B tokens.**

### Real sources
- FineWeb-Edu (primary, quality-filtered web)
- Proof-Pile-2 (math + CS papers)
- AlgebraicStack (verified math code)
- MathCode-Pile (math + code)
- OpenWebMath (web math content)
- Big-Math (>250k verified problems)
- MathPile (curated math corpus)
- AutoMathText V2 (`OpenSQZ/AutoMathText-V2`) -- quality-filtered math+STEM, 2.46T tokens available. LLM-scored quality (not just perplexity/keyword). (CR3-C2)

### DeepSeekMath-style CC Mining (CR3-C1)

Iterative fastText classifier to mine high-quality STEM content from Common Crawl, producing 5-9x more math tokens from the same source as OpenWebMath.

Pipeline:
1. Round 1: positives=OpenWebMath samples, negatives=random CC pages, train fastText, keep top 5% confidence
2. Round 2: human review top-yield URL patterns/domains, add confirmed domains as positive seeds, retrain, keep top 5%
3. Round 3: refine with hard negatives, MD5 dedup + benchmark decontamination

Target: expand real data before CLIMB-style clustering.

### Synthetic types
- Math+code hybrids (derivation + Python verification)
- Physics/chemistry simulation code
- First-principles derivations
- Cross-domain bridges
- Formal proof + intuition pairs
- Process-supervised step traces
- Competition math solutions
- Socratic chains
- Rephrased versions
- Paradox resolutions
- Synthetic reasoning traces from open teacher models

### PRM-Verified Synthetic Data Filter (CR3-C3)

Before adding ANY synthetic training example, verify EVERY reasoning step using a Process Reward Model (not just final answer). Keep example only if all steps score above threshold (e.g., 0.7).

PRM options: math-shepherd-mistral-7b-prm (open source), Qwen-Math-PRM variants, or Claude API as lightweight proxy for seed examples.

Apply to: all synthetic data types, especially first-principles derivations and competition math solutions. Not needed for real data.

### Complete Problem-Solution Pairs (CR3-C4)

NEVER split a problem-solution pair across sequence boundaries. Each training sequence must contain complete reasoning trajectories. Non-reasoning docs can split normally.

### Pipeline
1. **CLIMB-style semantic clustering**: embed all real data with small encoder, K-means into 20-25 clusters, train proxy to find optimal cluster weights (apply to FineWeb-Edu)
2. **MATES**: BERT-base proxy computes influence scores every 500 steps during pretraining, dynamically upweights high-influence data
3. **RHO-1**: selective token weighting from step 0 (not fine-tuning only). Train reference model, upweight tokens where excess loss is high.

### What's NOT in pretraining
- Lean 4 formal proofs: no measurable cross-domain transfer to informal STEM at <1B scale. Post-training only if at all.

---

## 5. Training Config

```python
# Optimizer: Three-group setup
# Group 1 - SSM params:     AdamW (lr=3e-4, betas=(0.9, 0.95), wd=0.1)
# Group 2 - Matrix params:  NorMuon + Polar Express (lr=0.02, momentum=0.95)
# Group 3 - Embed/1D/gates: AdamW (lr=3e-4, wd=0.1, grad_accum=2x)
# All groups: WSD schedule, per-group gradient clipping max_norm=1.0
# Cautious Weight Decay tied to LR on all groups.

# Schedule: WSD
warmup_steps = 1500        # 2% of 76,000 steps
stable_steps = 63000       # 83%
decay_steps = 11500        # 15%, linear to 0.1x peak LR

# Scale
seq_len = 4096
batch_size = 8
gradient_accumulation = 32  # effective batch = 256 sequences
tokens_per_step = 1_048_576
total = 80B tokens -> ~76,000 steps -> ~50-70 days on RTX 5070

# Training objectives (all active simultaneously):
loss = (
    ntp_loss                          # standard next-token prediction
    + 0.3 * mtp_loss                  # MTP-4 sequential heads
    + 0.1 * fsp_loss                  # Future Summary Prediction, window=64
    + rho1_selective_weighting        # excess loss token weighting
    + 0.05 * process_supervision_loss # step-level correctness signal
)
```

FA2 note: FA3 requires Hopper (sm_90). RTX 5070 is Blackwell consumer (sm_120) -- FA2 only as of March 2026. FP8 for matmuls, BF16 for SSM states.

### Polar Express (CR3-B1)

Drop-in replacement for Newton-Schulz 5 in Muon's orthogonalization step. Better polynomial approximation for polar decomposition in bfloat16. Lower final loss, same compute cost.

Action required: pull exact coefficients from modded-nanogpt PR #172 / record 38 commit.

### Cautious Weight Decay with LR-Tied Schedule (CR3-B2)

Weight decay scales with current learning rate ratio, not fixed. During WSD decay phase, weight decay drops proportionally:

```
wd_scale = current_lr / peak_lr   # 1.0 at peak, ~0.1 at end
effective_wd = base_wd * wd_scale  # 0.1 * ratio
```

Apply to all optimizer groups. Prevents over-regularization during final fine-grained adjustments.

### Embedding-Specific Gradient Accumulation (CR3-A5)

Embedding matrix and lm_head accumulate gradients for 2 optimizer steps before updating. Rest of network updates every step. Embedding gradients are high-variance (each token appears infrequently); accumulating gives more stable gradient estimates.

### Progressive Embed/lm_head Untying (CR2)

Gradual untying of embedding and lm_head weights during training. Untie at 2/3 of training.

---

## 6. Post-Training

### Stage 1: Evaluate Base Model (CR3-D1)

**Mandatory pre-RL quality threshold.** Before starting ANY post-training, evaluate the base pretrained model:

```
GSM8K zero-shot:  > 20%   (grade school math)
MATH-500:         > 10%   (competition math)
ARC-Challenge:    > 40%   (science reasoning)

If any metric is BELOW these thresholds:
  -> Do NOT start post-training
  -> Return to pretraining: add more math/reasoning data, train more
  -> Re-evaluate after 10B more tokens

These are MINIMUMS for post-training to be useful.
Below these: RL produces length bloat, not reasoning.
```

### Stage 2: Cold-Start SFT (MANDATORY -- 1B < 1.5B threshold)

Cold-start SFT is NOT optional at 1B -- it is mandatory. R1-Zero style (pure RL from scratch) does NOT work below 1.5B. It produces length bloat and instability.

- 5,000-10,000 curated long-CoT examples (human-refined, first-person style)
- Sources: OpenMathReasoning, OpenCodeReasoning, OpenScienceReasoning (subsets)
- **Magpie-style self-synthesis (CR3-C5)**: prompt Claude with STEM system prompt + empty user turn to generate both question AND answer
- **Include false starts and self-corrections (CR3-D2)**: 60% traces with self-correction, 40% clean correct traces
- Format: `<think>...reasoning with false starts...</think><answer>...</answer>`
- -> CogCore-1B-Thinking

### Stage 3: CoT Trajectory Distillation from Large Teacher (CRITICAL)

This is the single highest-impact post-training technique. MobileLLM-R1 proves 950M can hit MATH 74% through teacher distillation.

- Teacher: Qwen3-32B or DeepSeek-R1 (whichever is most accessible)
- Generate 500K-800K reasoning traces via rejection sampling + verification
- SFT on traces (standard cross-entropy)
- On-policy refinement: student generates -> align with teacher logits (GKD/on-policy KD)
- Expected gain: +10-35 points on hard benchmarks
- -> CogCore-1B-Distilled

### Stage 4: GRPO Reasoning RL

**Dr. GRPO configuration -- the empirically validated subset for <1.5B models.**

Full DAPO hurts at <1.5B (Qwen2.5-0.5B case study: Clip-Higher and dynamic sampling specifically degrade performance).

```
KEEP (Dr. GRPO):
  - Token-level policy gradient loss (not sample-level)
  - Overlong filtering (mask reward on truncated responses)
  - No KL divergence penalty (rely on clipping)
  - Symmetric clipping: eps = 0.2

REMOVED (hurts at <1.5B):
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
- 50K-100K math/code/STEM prompts
- Multi-stage: reasoning RL -> general/alignment RL
- -> CogCore-1B-Reasoner

### Stage 5: Tool Use SFT (NEW)

- 50K JSON/schema/tool trajectory examples
- Agentic RL with execution feedback
- Goal: reliable structured function calling
- -> CogCore-1B-Agent

### Stage 6: Merge + Evaluation

- SLERP merge of SFT + RL checkpoints (mandatory, CR3-D4)
- TIES/DARE for multi-stage conflict resolution
- Best-of-32 sampling with PRM scoring
- -> CogCore-1B-Final

### Inference: Best-of-N Sampling (CR3-D5)

Best-of-32 sampling with verifier at inference time. Highest-ROI test-time compute method for 1B.

- N=32: biggest relative gains (scaling curve steeper than large models, plateaus ~64-128)
- Temperature=0.8, max_tokens=4096, 1000+ token extended thinking
- Use Verifiable Reward Gym functions as verifier
- Fallback: model self-scoring for open-ended problems
- Alone gives 15-25% relative improvement on MATH/AIME vs greedy decoding

---

## 7. Phase Checklist

### Phase 1: BPE Tokenizer
- [x] BPE vocab=8192 (done, will retrain)
- [ ] Retrain tokenizer to vocab=128000 (CR5-C3) -- BLOCKS Phase 2
- [ ] Verify: vocab=128000, compression >= 15x bytes, digits split, roundtrip clean
- [ ] Verify STEM tokens: \frac, \partial, \int, \sum -> 1 token each
- [ ] Verify trilingual: Serbian Cyrillic, German umlauts -> single tokens
- [ ] Verify code tokens: common keywords, operators -> single tokens

### Phase 2: Data Pipeline
- [ ] Implement CLIMB-style semantic clustering on FineWeb-Edu
- [ ] Source and filter code/math/science subsets
- [ ] Generate synthetic reasoning traces from open teacher models
- [ ] Validate domain ratios on 1B-token ablation
- [ ] DeepSeekMath-style CC mining with fastText classifier (CR3-C1)
- [ ] AutoMathText V2 download and quality-score sampling (CR3-C2)
- [ ] Synthetic data generation (all types)
- [ ] PRM verification of all synthetic examples (CR3-C3)
- [ ] Document packer with complete problem-solution pairs (CR3-C4)
- [ ] Quality filtering + dedup

### Phase 3: Architecture (implement in order)
 1. Dense SwiGLU MLP (hidden=3392, zero-init)
 2. Mamba-3 MIMO SSMBlock (state=1280, d_conv=4, expand=2)
 3. SSMBlock = RMSNorm + Mamba-3 + RMSNorm + SwiGLU -> stack 26 layers
 4. GQA attention (20Q/4KV, d_head=64, QK-Norm, RoPE)
 5. HybridBlock = RMSNorm + parallel(Mamba-3, GQA) + gated-add + RMSNorm + SwiGLU
 6. Interleaved stack: 26xSSM + 6xHybrid at [4,9,14,19,24,29]
 7. Embedding projection (128000->640->1280) + tied LM head
 8. Verify ~1.03B params, all active
 9. Long-short attention (full at 4,14,24; local w=512 at 9,19,29)
10. Logit softcap=15
11. Zero-init all output projections
12. Gated value embeddings (hybrid blocks)
13. Gated U-net skip connections
14. Bigram hash embeddings (640->1280 projection)
15. Smear token 1-position
16. Partial key offset on GQA layers
17. Sparse attention gate on local-window layers (CR3-A3)
18. Backout mechanism at layer 21 (CR3-A2)
19. Remove post-attention lambdas (CR4-A5)
20. Paired head attention on local-window layers (CR3-A1)
21. Partitioned hyperconnections (CR3-A4) -- LAST

### Phase 4: Training Infrastructure
- [ ] SSM optimizer group: AdamW for Mamba-3 params
- [ ] NorMuon + Polar Express for matrix params
- [ ] Separate gradient clipping per group
- [ ] WSD schedule (warmup 2%, stable 83%, decay 15%)
- [ ] Cautious weight decay with LR schedule (CR3-B2)
- [ ] Embedding-specific gradient accumulation (CR3-A5)
- [ ] Progressive embed/lm_head untying (CR2)
- [ ] MTP-4 + FSP-64 heads
- [ ] RHO-1 selective weighting from step 0
- [ ] Process supervision loss

### Phase 5: Pretraining Run
- [ ] 80B tokens, ~50-70 days on RTX 5070
- [ ] MATES dynamic data selection
- [ ] EoS document alignment (CR2)
- [ ] Monitoring: loss, gradients, attention entropy
- [ ] -> CogCore-1B-Base

### Phase 6: Post-Training
- [ ] Evaluate base model: GSM8K>20%, MATH>10%, ARC>40% (CR3-D1)
- [ ] Cold-start SFT with Magpie-style generation (5-10K long-CoT)
- [ ] CoT trajectory distillation from large teacher (500-800K traces)
- [ ] On-policy KD refinement (GKD)
- [ ] Dr. GRPO reasoning RL (50-100K math/code/STEM prompts)
- [ ] General alignment RL
- [ ] Tool use SFT (50K examples)
- [ ] SLERP merge + TIES/DARE conflict resolution (mandatory)
- [ ] Best-of-32 evaluation + PRM scoring

---

## 8. Technique Table

| Technique | Source | Phase | Expected Gain |
|-----------|--------|-------|--------------|
| BPE 128000 + embed projection | Qwen3.5/GLM-5/MobileLLM-R1 | 1 | 15-20x byte compression |
| DeepSeekMath CC mining | arXiv:2402.03300 | 2 | 5-9x more math tokens |
| AutoMathText V2 | arXiv:2402.07625 | 2 | Quality-filtered 2.46T pool |
| CLIMB-style clustering | NVIDIA 2025 | 2 | 2-3x data efficiency |
| PRM-verified synthetic filter | MathShepherd 2025 | 2 | +2pp math reasoning |
| Complete sequence packing | Multiple 2025-2026 | 2 | Data hygiene, coherent chains |
| MATES dynamic selection | NeurIPS 2024 | 5 | +1.1-1.3% zero-shot |
| RHO-1 pretraining | RHO-1 paper | 4 | +6.8% avg 15 tasks |
| Process-supervised traces | Rationalyst 2025 | 2 | +3.9% generalization |
| GQA 20Q/4KV | Qwen/DeepSeek/Meta/NVIDIA | 3 | Battle-tested at 0.8-1.5B |
| Interleaved hybrid (5:1 SSM:attn) | Mamba-3 paper / Hymba | 3 | 40% less VRAM vs all-attention |
| Mamba-3 MIMO | ICLR 2026 | 3 | +0.6-1.8pp over DeltaNet |
| Gated addition merge | Hymba 2025 | 3 | Proven, simple SSM+attn merge |
| Dense SwiGLU (no MoE) | March 2026 consensus | 3 | 100% param utilization |
| QK-Norm | OLMo2/Gemma3 | 3 | Stability, free |
| nGPT unit norms | arXiv 2410.01131 | 3 | 4x faster convergence |
| SkipV1 connections | NeurIPS 2025 | 3 | 25-50% KV-cache reduction |
| Long-short attention | Addendum A | 3 | Efficient local+global mix |
| Logit softcap=15 | CR2 | 3 | Stability |
| Zero-init output proj | CR2 | 3 | Clean init |
| Gated value embeds/skips | modded-nanogpt rec 55 | 3 | Learned injection control |
| Bigram hash embeddings | Addendum A | 3 | Subword context |
| Smear token 1-position | Addendum A | 3 | Position smoothing |
| Partial key offset | modded-nanogpt rec 49 | 3 | Positional look-ahead |
| Paired head attention | modded-nanogpt rec 58 | 3 | Richer position mixing |
| Backout mechanism | modded-nanogpt rec ~40 | 3 | Cleaner predictions |
| Sparse attention gate | modded-nanogpt rec 28 | 3 | Prevents attention sink |
| Partitioned hyperconnections | modded-nanogpt rec 73 | 3 | Independent read paths |
| NorMuon + Polar Express | arXiv:2510.05491 / arXiv:2505.16932 | 4 | Better polar decomp in bf16 |
| Cautious WD schedule | modded-nanogpt rec 43+50 | 4 | Less over-regularization |
| Embed gradient accum | modded-nanogpt | 4 | Stable embed updates |
| Progressive embed untying | CR2 | 4 | Gradual capacity split |
| WSD schedule | DeepSeek V3 / Kimi K2 | 4 | Best schedule |
| MTP-4 (0.3) + FSP-64 (0.1) | DeepSeek/Meta | 4 | Multi-objective |
| Pre-RL quality threshold | Research report Q5 | 6 | Gate bad RL starts |
| Cold-start CoT (mandatory) | Multiple 2025-2026 | 6 | Required for <1.5B RL |
| Magpie-style self-synthesis | ICLR 2025 | 6 | Diverse CoT generation |
| False starts in CoT | Research report Q8 | 6 | Error recovery learning |
| CoT trajectory distillation | MobileLLM-R1 / DeepSeek-R1 | 6 | +10-35pp hard benchmarks |
| Tool use SFT | Qwen3.5 / Kimi K2 | 6 | Reliable function calling |
| Dr. GRPO (stripped DAPO) | ByteDance/Qwen2.5 study | 6 | Same result, fewer steps |
| Verifiable Reward Gym | Kimi K2 methodology | 6 | Multi-domain RL rewards |
| SLERP merge (mandatory) | Research report Q13 | 6 | Recover lost knowledge |
| TIES/DARE merge | Multi-stage RL merge | 6 | Conflict resolution |
| Best-of-32 inference | Research report Q12 | 6 | 15-25% relative improvement |

## 9. Expected Outcomes

Target: **7-10B dense transformer equivalent** on STEM reasoning, with ceiling of 12B on targeted math/code benchmarks via test-time compute (Best-of-32 + PRM).

Baseline reference points (March 2026 1B-class SOTA):
- MobileLLM-R1-950M: MATH 74%, GSM8K 67.5%, AIME 15-40%, HumanEval 46%
- Hymba-1.5B: MMLU 52.8%, GSM8K 58.8%
- Qwen3.5-0.8B: Strong multimodal/edge but lower absolute scores

CogCore-1B targets (post-training complete):
```
MMLU: 60-70%
GSM8K: 80-90%
MATH: 50-75% (depending on distillation quality)
HumanEval: 50-60%
ARC-Challenge: 80-88%
```

These would place CogCore-1B at or above Hymba-1.5B and competitive with MobileLLM-R1 on reasoning, while being architecturally distinct (Mamba-3 hybrid vs pure transformer).

## What NOT to Do at This Scale

| Technique | Why Skip |
|-----------|----------|
| DiffTrans at any scale | Zero frontier adoption, niche research dead end |
| MLA below 7B | Overhead > benefit, GQA wins |
| Vocab < 65K | 2026 minimum is 65K, 128K+ standard |
| MoE at training time (<1B) | Routing overhead eats gains, 100% utilization with dense |
| Full DAPO (asymmetric clip, dynamic sampling) | Hurts at <1.5B (Qwen2.5-0.5B case study) |
| R1-Zero style (pure RL, no cold-start) | Length bloat, no "aha moment" below 1.5B |
| Skip distillation | Single highest-impact post-training technique |
| Assume FA3 on consumer Blackwell | sm_120 = FA2 only as of March 2026 |
| Trust 35-day training estimate | Realistic: 50-70 days on RTX 5070 |
| RLHF with human raters | Cost of reward model > value at this scale |
| Multi-agent reasoning | Context too small, model too weak to self-critique |
| RAG | Retrieval overhead >> model itself |
| Lean 4 in pretraining | No cross-domain transfer at <1B |
| Constitutional AI | Confirmed non-viable at <1B -- permanently dropped |

## What Remains for CR6

Pending further research or implementation experience:
- Exact CLIMB-style data curation (pending ClimbMix public release)
- MoE-upcycling AFTER pretraining
- Extended context beyond 4096 (YaRN/NTK for 8K-32K)
- Speculative decoding (EAGLE-style self-speculation)
- Self-play data bootstrapping
- FAC-SAE diversity metrics
- Mamba-3 -> Gated DeltaNet fallback path
- Exact Polar Express polynomial coefficients
- Evaluate Qwen3.5 Gated DeltaNet pattern as alternative architecture

---

*All changes backed by 2025-2026 empirical results at <=1B scale.*
*Primary sources: Mamba-3 (ICLR 2026), Hymba (NVIDIA 2025), Falcon-H1 (arXiv 2507.22448), MobileLLM-R1 (Meta Feb 2026), Qwen 3.5 (Feb 2026), DeepSeek-R1 (2025), NorMuon (arXiv:2510.05491), Polar Express (arXiv:2505.16932), Kimi K2 (Moonshot AI 2026), NVIDIA Nemotron-CLIMBMix (2026), nanochat (Karpathy Mar 2026), ClimbMix (NVIDIA 2025), MATES (NeurIPS 2024), DAPO (ByteDance 2503.14476), E2H Reasoner (2506.06632), Rationalyst (2025), SkipV1Former (NeurIPS 2025), nGPT (arXiv 2410.01131), modded-nanogpt PRs #117/#130/#140/#172/#191/#230 + records #38-74, DeepSeekMath (arXiv:2402.03300), AutoDS (ACL 2025), MathShepherd (2025), Qwen3 technical report (2025).*