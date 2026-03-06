# Current Step: Pre-Pipeline Setup -- Sovereign Qwen 3.5 9B

**Status: NOT STARTED**

**Previous: Strategic pivot from CogCore-1B to Sovereign Qwen 3.5 9B post-training**

## Why This Step

Before any training begins, we need a working environment with verified tooling, baseline benchmarks, and a confirmed QLoRA training loop. Every subsequent phase depends on this foundation.

## Sub-tasks

### Environment Setup
- [ ] Install Unsloth (latest, with Qwen 3.5 support)
- [ ] Install PEFT (latest)
- [ ] Install TRL (latest, GRPOTrainer)
- [ ] Build llama.cpp from source (`-DCMAKE_CUDA_ARCHITECTURES=120`)
- [ ] Install Ollama (latest)
- [ ] Verify torch.compile works on WSL2/Linux (10-30% speedup via Triton)
- [ ] Setup vLLM for local inference

### Model Download
- [ ] Download Qwen 3.5 9B from HuggingFace (full precision for training)
- [ ] Download Q4_K_M GGUF for inference testing

### Baseline Benchmarking
- [ ] MMLU-Pro (stock target: 82.5)
- [ ] GPQA Diamond (stock target: 81.7)
- [ ] HMMT Feb 25 (stock target: 83.2)
- [ ] GSM8K
- [ ] MATH-500
- [ ] HumanEval
- [ ] ARC-Challenge
- [ ] BFCL-V4 (stock target: 66.1)
- [ ] TAU2-Bench (stock target: 79.1)
- [ ] IFEval (stock target: 91.5)

### QLoRA Test Run
- [ ] Run small QLoRA training (100 examples, 5 minutes) -- verify no OOM, loss decreases
- [ ] Test DoRA + NEFTune + rsLoRA flags work with Unsloth

### Cloud Prep
- [ ] Create account on Lambda/RunPod
- [ ] Verify 8xH100 availability and pricing
- [ ] Test SSH + data transfer workflow

## Acceptance Criteria

1. Model loads successfully on RTX 5070 (NF4 quantized)
2. QLoRA training works without OOM at seq_len=4096
3. Baseline benchmarks recorded for all 10 suites
4. Inference speed >= 80 t/s at Q4_K_M via llama.cpp
5. Cloud rental account ready with verified access
6. torch.compile status confirmed (WSL2/Linux vs Windows)

## What Comes Next

Phase 1 (Vocab Surgery, optional) or Phase 2 (Domain-Adaptive CPT on rented 8xH100).
