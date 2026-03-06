# Current Step: Pre-Pipeline Setup -- Sovereign Qwen 3.5 9B

**Status: IN PROGRESS**

**Previous: Strategic pivot from CogCore-1B to Sovereign Qwen 3.5 9B post-training**

## Why This Step

Before any training begins, we need baseline benchmarks on the stock model, a working eval harness, and verified tooling. Every subsequent phase compares against these baseline numbers.

## Sub-tasks

### Model Download + Inference
- [x] Download Qwen 3.5 9B GGUF (at `C:\Users\djord\.lmstudio\models\lmstudio-community\Qwen3.5-9B-GGUF`)
- [x] Verify inference speed (75-85 t/s in LM Studio, meets 80+ t/s target)
- [x] Setup llama-server with CUDA 13.1 (b8215, native sm_120 Blackwell kernels)
- [x] Verify llama-server speed: 85-92 t/s generation, 170+ t/s prompt processing
- [ ] Download HF weights for training (full precision, needed for QLoRA)

### Evaluation Setup
- [x] Install lm-evaluation-harness v0.4.11 in `.venv-eval/` (`pip install "lm_eval[ifeval,math,api]"`)
- [x] Configure llama-server OpenAI-compatible API at localhost:8080
- [x] HuggingFace auth configured (for gated datasets like GPQA)
- [x] Create `results/baseline/` directory for scores
- [x] Verified pipeline end-to-end: IFEval 2-sample test passed, results saved correctly
- [ ] Run Baseline Benchmarks -- **THE MAIN TASK (NEXT SESSION)**
  - [ ] GPQA Diamond (stock target: 81.7) -- `gpqa_diamond_cot_zeroshot`, 198 Qs, ~1-2 hrs
  - [ ] IFEval (stock target: 91.5) -- `ifeval`, 541 Qs, ~2-4 hrs
  - [ ] AIME 2025 (~40-55 est.) -- `aime25`, 30 Qs, ~1-2 hrs, use `max_gen_toks=8192`
  - [ ] MMLU-Pro (stock target: 82.5) -- needs generative variant or different backend (deferred)
  - [ ] SuperGPQA (stock target: 58.2) -- separate repo, not in lm-eval (deferred)
  - [ ] LiveCodeBench v6 (stock target: 65.6) -- separate repo + Docker (deferred)
  - [ ] BFCL-V4 (stock target: 66.1) -- separate repo + Docker (deferred)
  - [ ] TAU2-Bench (stock target: 79.1) -- separate repo (deferred)
  - [ ] RULER (establish baseline at 4K/16K/64K/128K) -- long-context, may hit VRAM limits (deferred)
  - [ ] LongBench v2 (stock target: 55.2) -- needs extra pip deps (deferred)
- [ ] Record 20 forgetting canary problems (after baselines reveal model strengths/weaknesses)

### Environment Setup
- [ ] Install Unsloth (latest, with Qwen 3.5 support)
- [ ] Install PEFT (latest)
- [ ] Install TRL (latest, GRPOTrainer)
- [x] llama.cpp pre-built CUDA 13.1 (no need to build from source)
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

## Benchmarking Setup (Verified Working)

### llama-server location
`C:\AI\llama-cpp-server\` -- llama.cpp b8215, CUDA 13.1, pre-built binaries

### Server start command
```bash
cd C:\AI\llama-cpp-server
./llama-server.exe \
  -m "C:/Users/djord/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf" \
  --port 8080 --n-gpu-layers 99 --ctx-size 16384 --parallel 2 \
  --flash-attn on --jinja --host 127.0.0.1 --reasoning-format none
```

### Benchmark command template
```bash
cd C:\AI\NeuralNetZero
source .venv-eval/Scripts/activate
PYTHONIOENCODING=utf-8 python -m lm_eval run \
  --model local-chat-completions \
  --model_args "model=Qwen3.5-9B-Q4_K_M.gguf,base_url=http://localhost:8080/v1/chat/completions,num_concurrent=2,max_gen_toks=4096" \
  --tasks <TASK_NAME> \
  --batch_size 1 --apply_chat_template \
  --output_path results/baseline/ --log_samples
```

### Known issues resolved
- `--flash-attn` requires explicit `on` value (not bare flag) in b8215
- `--apply_chat_template` required for `local-chat-completions` backend
- `PYTHONIOENCODING=utf-8` needed on Windows to print Unicode results table
- GPQA is gated dataset -- HF auth configured
- Thinking mode: `--reasoning-format none` keeps thinking in `content` (visible to lm-eval regex filters)
- Context exceeded at 8K with 4 parallel slots -> use 16K ctx with 2 parallel slots

## Acceptance Criteria

1. All available Sovereign benchmarks scored at deployment quant (Q4_K_M)
2. Results saved in `results/baseline/`
3. 20 forgetting canary problems established
4. QLoRA training verified (no OOM at seq_len=4096)
5. Cloud rental account ready
6. torch.compile status confirmed (WSL2/Linux vs Windows)

## What Comes Next

Phase 1 (Vocab Surgery, optional) or Phase 2 (Domain-Adaptive CPT on rented 8xH100).
