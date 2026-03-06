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

Normalization:
  Pre-RMSNorm on residual stream
  QK-Norm on Q and K
  nGPT unit norms on embedding + output weight matrices

Other:
  No weight tying, no biases
  BPE 8192, context 2048

Total params: ~490M sparse, ~350M active at inference
```

### Why these choices

- **DiffTrans over standard MHA**: +7.5% avg reasoning, needs only 65% params to match baseline (ICLR 2025)
- **Mamba2 hybrid (6/32 layers)**: +2% ARC, 2-8x faster inference on long contexts
- **32L x d=896 over 24L x d=1024**: deeper = more sequential reasoning steps, same param count
- **QK-Norm**: adopted by OLMo2, Gemma3, every serious 2025 small model. Zero overhead.
- **nGPT unit norms**: 4x faster convergence, reduces warmup needed
- **SkipV1**: 25-50% KV-cache reduction + better perplexity via first-layer V reuse

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

### Pipeline
1. **ClimbMix**: embed all real data with small encoder, K-means into 20-25 clusters, train 50M proxy to find optimal cluster weights
2. **MATES**: BERT-base proxy computes influence scores every 500 steps during pretraining, dynamically upweights high-influence data
3. **RHO-1**: selective token weighting from step 0 (not fine-tuning only). Train reference model, upweight tokens where excess loss is high.

### What's NOT in pretraining
- Lean 4 formal proofs: no measurable cross-domain transfer to informal STEM at <1B scale. Post-training only if at all.

---

## 5. Training Config

```python
# Optimizer: Cautious 8-bit Muon + MuonClip
optimizer = Cautious_8bit_Muon(
    matrix_params=[...],  # all 2D weights except embed/norm
    lr=0.02,
    momentum=0.95,
    weight_decay=0.1,
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

---

## 6. Post-Training

### Stage 1: Cold-Start CoT
- Generate 5,000 Chain-of-Thought traces using frontier model
- Format: `<think>...reasoning...</think><answer>...</answer>`
- SFT fine-tune -> CogCore-500M-Thinking

### Stage 2: DAPO Reasoning RL
- DAPO (not GRPO) as primary RL algorithm
- On Qwen2.5-32B: 50 AIME points vs GRPO's 30 -- same gains in 50% fewer steps
- Four core changes:
  1. Asymmetric clipping: eps_high=0.28, eps_low=0.2
  2. Dynamic sampling: remove zero-variance prompts, oversample mixed-reward
  3. Token-level policy gradient (prevents long-sequence reward dilution)
  4. No KL divergence penalty (rely on clipping)
- E2H curriculum: easy-to-hard problem scheduling
- 2,000-5,000 RL steps -> CogCore-500M-Reasoner

### Stage 3: On-Policy Self-Distillation
- Student generates 8 responses per problem
- Score with rule-based verifier or teacher model
- SFT on top-k responses, repeat 3-5 rounds
- On-policy dominates off-policy by 4-8x token efficiency
- -> CogCore-500M-Final

### Stage 4: Test-Time Scaling
- Train for long-horizon verifiable reasoning
- Reward long correct traces more than short ones
- Inference: Best-of-N sampling with lightweight verifier

---

## 7. Phase Checklist

### Phase 1: BPE Tokenizer
- [ ] HuggingFace `tokenizers` library, BPE vocab=8192
- [ ] `individual_digits=True`, byte-level fallback
- [ ] Train on STEM corpus (existing data + synthetic templates)
- [ ] Integrate into codebase, backward-compatible with char tokenizer

### Phase 2: Data Pipeline
- [ ] ClimbMix semantic clustering (20-25 clusters)
- [ ] Synthetic data generation (11 types)
- [ ] Document packing (DeepSeek-style)
- [ ] Quality filtering + dedup

### Phase 3: Architecture
- [ ] DiffTrans attention + MLA
- [ ] Mamba2 hybrid layers (6/32)
- [ ] 32L x d=896 dimensions
- [ ] QK-Norm, nGPT unit norms
- [ ] MoE SwiGLU (8 experts, top-2)
- [ ] SkipV1 connections
- [ ] Gradient checkpointing

### Phase 4: Training Infrastructure
- [ ] Cautious 8-bit Muon optimizer
- [ ] WSD schedule (warmup=500)
- [ ] MTP-4 + FSP-64 heads
- [ ] RHO-1 selective weighting from step 0
- [ ] Process supervision loss

### Phase 5: Pretraining Run
- [ ] 80B tokens, ~16 days on RTX 5070
- [ ] MATES dynamic data selection
- [ ] Monitoring: loss, gradients, attention entropy
- [ ] -> CogCore-500M-Base

### Phase 6: Post-Training
- [ ] Cold-start CoT (5,000 traces)
- [ ] DAPO + E2H curriculum
- [ ] On-policy self-distillation
- [ ] Test-time scaling setup

---

## 8. Technique Table

| Technique | Source | Phase | Expected Gain |
|-----------|--------|-------|--------------|
| BPE 8192 (STEM-tuned) | All 2026 models | 1 | 3-5x compression |
| ClimbMix clustering | NVIDIA 2025 | 2 | 2-3x data efficiency |
| MATES dynamic selection | NeurIPS 2024 | 5 | +1.1-1.3% zero-shot |
| RHO-1 pretraining | RHO-1 paper | 4 | +6.8% avg 15 tasks |
| Process-supervised traces | Rationalyst 2025 | 2 | +3.9% generalization |
| DiffTrans attention | ICLR 2025 | 3 | +7.5% reasoning |
| Mamba2 hybrid | Nemotron-H | 3 | +2% ARC, 2-8x inference |
| 32L x d=896 | SmolLM3/IMU-1 | 3 | +1-3% hard reasoning |
| QK-Norm | OLMo2/Gemma3 | 3 | Stability, free |
| nGPT unit norms | arXiv 2410.01131 | 3 | 4x faster convergence |
| SkipV1 connections | NeurIPS 2025 | 3 | 25-50% KV-cache reduction |
| Cautious 8-bit Muon | arXiv 2411.16085 | 4 | Measurable perplexity |
| WSD schedule | DeepSeek V3 | 4 | Best schedule |
| MTP-4 (0.3) + FSP-64 (0.1) | DeepSeek/Meta | 4 | Multi-objective |
| DAPO | ByteDance 2503.14476 | 6 | Same result in 50% steps |
| E2H curriculum | arXiv 2506.06632 | 6 | Generalization boost |
| On-policy distillation | 2025-2026 trend | 6 | 4-8x token efficiency |

## 9. Expected Outcomes

| Metric | 1M (done) | After BPE + data | CogCore-500M | After DAPO |
|--------|-----------|------------------|--------------|------------|
| Loss | ~2.5 (chars) | ~1.8 (BPE) | ~1.0 | <1.0 |
| Coherent STEM answers | No | Partial | Yes | Yes, verified |
| Multi-step reasoning | No | No | Yes | Yes, improved |
| Follows instructions | No | No | Partial | Yes |
| Domain transfer | None | Weak | Moderate | Strong |

**Net result:** A 500M model that performs like a 1.5-2B standard transformer on STEM reasoning. The IQ-first, facts-from-tools framing is the correct goal.

## What NOT to Do at This Scale

| Technique | Why Skip |
|-----------|----------|
| RLHF with human raters | Cost of reward model > value at this scale |
| Multi-agent reasoning | Context too small, model too weak to self-critique |
| RAG | Retrieval overhead >> model itself |
| FP8 training | <5% gain, high complexity |
| Lean 4 in pretraining | No cross-domain transfer at <1B |

---

*All changes backed by 2025-2026 empirical results at <1B scale.*
*Primary sources: DiffTrans (ICLR 2025), Nemotron-H (2504.03624), ClimbMix (NVIDIA 2025), MATES (NeurIPS 2024), DAPO (ByteDance 2503.14476), Cautious-Adam (arXiv 2411.16085), 8-bit Muon (2509.23106), E2H Reasoner (2506.06632), Rationalyst (2025), SkipV1Former (NeurIPS 2025), nGPT (arXiv 2410.01131).*
