# Current Step: Pre-Pipeline Setup -- Sovereign Qwen 3.5 9B

**Status: IN PROGRESS**

**Previous: Strategic pivot from CogCore-1B to Sovereign Qwen 3.5 9B post-training**

## Why This Step

Before any training begins, we need baseline benchmarks on the stock model, a working eval harness, and verified tooling. Every subsequent phase compares against these baseline numbers.

## Sub-tasks

### Model Download + Inference
- [x] Download Qwen 3.5 9B GGUF (at `C:\Users\djord\.lmstudio\models\lmstudio-community\Qwen3.5-9B-GGUF`)
- [x] Verify inference speed (75-85 t/s in LM Studio, meets 80+ t/s target)
- [ ] Download HF weights for training (full precision, needed for QLoRA)

### Evaluation Setup
- [ ] Install lm-evaluation-harness (`pip install "lm_eval[hf,vllm]"`)
- [ ] Configure LM Studio OpenAI-compatible API (already at localhost:1234)
- [ ] Run Full Eval (Sovereign 10) baseline -- **THE MAIN TASK**
  - [ ] GPQA Diamond (stock target: 81.7)
  - [ ] SuperGPQA (stock target: 58.2)
  - [ ] MMLU-Pro (stock target: 82.5)
  - [ ] AIME 2025 (~40-55 est.)
  - [ ] LiveCodeBench v6 (stock target: 65.6)
  - [ ] BFCL-V4 (stock target: 66.1)
  - [ ] TAU2-Bench (stock target: 79.1)
  - [ ] RULER (establish baseline at 4K/16K/64K/128K)
  - [ ] IFEval (stock target: 91.5)
  - [ ] LongBench v2 (stock target: 55.2)
- [ ] Record 20 forgetting canary problems (5 knowledge, 5 math, 5 code, 5 instruction following)
- [ ] Create `results/baseline/` directory for scores

### Environment Setup
- [ ] Install Unsloth (latest, with Qwen 3.5 support)
- [ ] Install PEFT (latest)
- [ ] Install TRL (latest, GRPOTrainer)
- [ ] Build llama.cpp from source (`-DCMAKE_CUDA_ARCHITECTURES=120`)
- [ ] Install Ollama (latest)
- [ ] Verify torch.compile works on WSL2/Linux
- [ ] Setup vLLM for local inference

### QLoRA Test Run
- [ ] Run small QLoRA training (100 examples, 5 minutes) -- verify no OOM, loss decreases
- [ ] Test DoRA + NEFTune + rsLoRA flags work with Unsloth

### Cloud Prep
- [ ] Create account on Lambda/RunPod
- [ ] Verify 8xH100 availability and pricing
- [ ] Test SSH + data transfer workflow

## Acceptance Criteria

1. All 10 Sovereign benchmarks scored at deployment quant (Q4_K_M)
2. Results saved in `results/baseline/`
3. 20 forgetting canary problems established
4. QLoRA training verified (no OOM at seq_len=4096)
5. Cloud rental account ready
6. torch.compile status confirmed (WSL2/Linux vs Windows)

## What Comes Next

Phase 1 (Vocab Surgery, optional) or Phase 2 (Domain-Adaptive CPT on rented 8xH100).
