# Current Step: Phase 2 -- Data Pipeline

**Status: NOT STARTED**

**Previous: Phase 1 (BPE Tokenizer) -- COMPLETE** (8192 vocab, 5.0x compression, all tests pass)

## Why This Phase

The model is only as good as its data. Phase 2 builds the infrastructure to collect, filter, and pack 80B tokens of high-quality STEM training data. CR3 data changes (C1-C5) are integrated here.

## Sub-tasks

### Real Data Collection
- [ ] Download AutoMathText V2 (`OpenSQZ/AutoMathText-V2`), sample top quality-score documents (CR3-C2)
- [ ] Set up DeepSeekMath-style CC mining pipeline with fastText classifier (CR3-C1)
  - [ ] Round 1: OpenWebMath positives vs random CC negatives
  - [ ] Round 2: domain seed refinement
  - [ ] Round 3: hard negatives + MD5 dedup + benchmark decontamination
- [ ] Download Proof-Pile-2, AlgebraicStack, MathCode-Pile, OpenWebMath, Big-Math, MathPile
- [ ] Quality filtering + global dedup across all sources

### Synthetic Data Generation
- [ ] Build generation scripts for 10 synthetic data types
- [ ] Magpie-style self-synthesis for cold-start CoT traces (CR3-C5)
- [ ] PRM verification pipeline for all synthetic examples (CR3-C3)
  - [ ] Integrate math-shepherd-mistral-7b-prm or equivalent
  - [ ] Verify every reasoning step, reject if any step < threshold

### Data Pipeline Infrastructure
- [ ] ClimbMix semantic clustering (20-25 clusters, 50M proxy for optimal weights)
- [ ] Document packer with complete problem-solution pair guarantee (CR3-C4)
  - [ ] Never split problem/derivation/proof across sequence boundaries
  - [ ] Non-reasoning docs can split normally
- [ ] MATES dynamic selection setup (BERT-base proxy, influence scores every 500 steps)
- [ ] RHO-1 reference model training for token-level weighting

## Acceptance Criteria

1. Real data sources downloaded and deduplicated
2. fastText CC mining classifier trained, first expansion run complete
3. Synthetic generation produces all 10 types with PRM filtering
4. ClimbMix clusters computed, optimal mixture weights found
5. Document packer verified: no problem-solution splits across boundaries
6. Total corpus size on track for 80B tokens
7. Data loading pipeline feeds batches to training loop

## What Comes Next

Phase 3: Architecture -- implement CogCore-500M model (15 architecture changes in order, QK-Norm first, Hyperconnections last).
