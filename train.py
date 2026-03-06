"""Training script for NeuralNetZero 1M param STEM LLM."""

import json
import math
import os
import sys
import time

import torch
from torch.utils.data import Dataset, DataLoader

from config import ModelConfig, TrainConfig
from tokenizer import CharTokenizer, BPETokenizer
from model import GPT
from optim import build_optimizer


# --- Data loading ---

def load_data(data_dir: str = "data") -> list[str]:
    """Load all .jsonl files from data directory."""
    texts = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    texts.append(obj["text"])
    return texts


class TextDataset(Dataset):
    """Concatenate all texts with EOS, chunk into block_size sequences."""

    def __init__(self, token_ids: list[int], block_size: int):
        self.block_size = block_size
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.n_chunks = max(1, (len(self.data) - 1) // block_size)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int):
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self.data[start:end]
        # Pad if needed
        if len(chunk) < self.block_size + 1:
            pad = torch.zeros(self.block_size + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        return chunk[:-1], chunk[1:]


# --- LR schedule ---

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# --- Text generation ---

@torch.no_grad()
def generate(model, tokenizer, prompt: str = "", max_len: int = 100,
             temperature: float = 0.8, top_k: int = 20,
             block_size: int = 512) -> str:
    model.eval()
    device = next(model.parameters()).device

    if prompt:
        ids = tokenizer.encode(prompt)
    else:
        ids = [tokenizer.bos_id]

    ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        idx_cond = ids[:, -block_size:]
        logits = model(idx_cond)[:, -1, :]
        logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

        if next_id.item() == tokenizer.eos_id:
            break

    model.train()
    return tokenizer.decode(ids[0].tolist())


# --- Loss bar ---

def loss_bar(loss_val: float, width: int = 30) -> str:
    filled = min(width, max(0, int(loss_val / 5.0 * width)))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


# --- Main ---

def main():
    tc = TrainConfig()

    # Device setup
    if tc.device == "cuda" and not torch.cuda.is_available():
        tc.device = "cpu"
        tc.dtype = "float32"
        tc.compile = False
        print("CUDA not available, falling back to CPU + float32")

    device = torch.device(tc.device)
    dtype = torch.bfloat16 if tc.dtype == "bfloat16" else torch.float32

    if tc.device == "cuda":
        # Check bfloat16 support
        if tc.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            dtype = torch.float16
            print("BF16 not supported, using FP16")

    print(f"Device: {device} | Dtype: {dtype}")

    # Load data
    print("\nLoading data...")
    texts = load_data()
    print(f"  {len(texts)} examples loaded")

    # Curriculum: sort by length (short first)
    texts.sort(key=len)

    # Build tokenizer
    mc = ModelConfig()
    if mc.tokenizer_type == "bpe" and mc.tokenizer_path:
        tokenizer = BPETokenizer(mc.tokenizer_path)
        print(f"  Tokenizer: BPE from {mc.tokenizer_path}")
    else:
        tokenizer = CharTokenizer(texts)
        print(f"  Tokenizer: character-level")
    print(f"  Vocab size: {tokenizer.vocab_size} tokens")

    # Tokenize all data with EOS separators
    all_ids = []
    for text in texts:
        all_ids.append(tokenizer.bos_id)
        all_ids.extend(tokenizer.encode(text))
        all_ids.append(tokenizer.eos_id)
    print(f"  Total tokens: {len(all_ids)}")

    # Train/val split (90/10)
    split = int(len(all_ids) * 0.9)
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    # Model config
    mc.vocab_size = tokenizer.vocab_size
    print(f"\n{'='*60}")
    print(f"MODEL ARCHITECTURE")
    print(f"{'='*60}")
    print(f"  Layers:     {mc.n_layer}")
    print(f"  d_model:    {mc.d_model}")
    print(f"  Heads:      {mc.n_head} (d_head={mc.d_head})")
    print(f"  FFN inner:  {mc.ffn_inner} (SwiGLU)")
    print(f"  Context:    {mc.block_size}")
    print(f"  Vocab:      {mc.vocab_size}")

    # Build model
    model = GPT(mc).to(device)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*60}")

    # Optional compile (requires Triton, not available on Windows)
    compiled = False
    if tc.compile and tc.device == "cuda" and sys.platform != "win32":
        try:
            model = torch.compile(model)
            compiled = True
            print("  torch.compile: enabled")
        except Exception as e:
            print(f"  torch.compile: failed ({e}), continuing without")
    if not compiled:
        print("  torch.compile: disabled")

    # Dataset + DataLoader
    block_size = mc.block_size
    train_dataset = TextDataset(train_ids, block_size)
    val_dataset = TextDataset(val_ids, block_size)
    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, shuffle=True,
                              drop_last=True, pin_memory=(tc.device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=tc.batch_size, shuffle=False,
                            drop_last=False)

    # Optimizer
    optimizers = build_optimizer(model, tc)
    print(f"  Optimizers: Muon (2D weights) + AdamW (embeddings/norms)")

    # Adjust max_steps if dataset is small
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        print("ERROR: Not enough data for a single batch. Reduce batch_size.")
        return
    total_epochs = max(1, tc.max_steps // steps_per_epoch)
    actual_max_steps = total_epochs * steps_per_epoch
    print(f"  Steps/epoch: {steps_per_epoch} | Epochs: {total_epochs} | Total steps: {actual_max_steps}")

    # Training
    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}")
    random_loss = math.log(tokenizer.vocab_size)
    print(f"  Random baseline loss: {random_loss:.2f}")
    print()

    model.train()
    step = 0
    train_losses = []
    val_losses = []
    t_start = time.time()

    for epoch in range(total_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # LR schedule — update all optimizers
            lr_adamw = get_lr(step, tc.warmup_steps, actual_max_steps, tc.learning_rate)
            lr_muon = get_lr(step, tc.warmup_steps, actual_max_steps, tc.muon_lr)
            for opt in optimizers:
                for pg in opt.param_groups:
                    if isinstance(opt, torch.optim.AdamW):
                        pg["lr"] = lr_adamw
                    else:
                        pg["lr"] = lr_muon

            # Forward + backward with autocast
            with torch.autocast(device_type=tc.device, dtype=dtype, enabled=(dtype != torch.float32)):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1),
                    ignore_index=tokenizer.pad_id,
                )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)

            for opt in optimizers:
                opt.step()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            train_losses.append(loss.item())

            # Logging
            elapsed = time.time() - t_start
            if step % 10 == 0:
                bar = loss_bar(loss.item())
                sys.stdout.write(f"\r  step {step+1:5d}/{actual_max_steps} | loss {loss.item():.4f} {bar} | lr {lr_adamw:.1e} | {elapsed:.0f}s")
                sys.stdout.flush()

            # Eval
            if step > 0 and step % tc.eval_interval == 0:
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        with torch.autocast(device_type=tc.device, dtype=dtype, enabled=(dtype != torch.float32)):
                            vlogits = model(vx)
                            vloss = torch.nn.functional.cross_entropy(
                                vlogits.view(-1, vlogits.size(-1)), vy.view(-1),
                                ignore_index=tokenizer.pad_id,
                            )
                        val_loss_sum += vloss.item() * vx.size(0)
                        val_count += vx.size(0)
                avg_val = val_loss_sum / max(1, val_count)
                val_losses.append((step, avg_val))
                print(f"\n  [eval] step {step} | val_loss {avg_val:.4f}")
                model.train()

            # Sample generation
            if step > 0 and step % tc.sample_interval == 0:
                sample = generate(model, tokenizer, prompt="", max_len=80, temperature=0.8, block_size=block_size)
                print(f"  [sample] \"{sample}\"")
                model.train()

            step += 1

    total_time = time.time() - t_start
    print(f"\n\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"  Final val loss:   {val_losses[-1][1]:.4f}")
    print(f"  Started at:       {train_losses[0]:.4f}")

    # ASCII loss curve
    print(f"\n{'='*60}")
    print(f"LOSS CURVE")
    print(f"{'='*60}")
    width = 60
    height = 15
    if len(train_losses) > width:
        sampled = [train_losses[int(i * len(train_losses) / width)] for i in range(width)]
    else:
        sampled = train_losses
    max_l = max(sampled)
    min_l = min(sampled)
    rng = max_l - min_l if max_l != min_l else 1
    for row in range(height):
        threshold = max_l - (row / (height - 1)) * rng
        line = ""
        for val in sampled:
            line += "#" if val >= threshold else " "
        label = f"{threshold:.2f}" if row % 3 == 0 else "     "
        print(f"  {label:>5s} |{line}|")
    print(f"        +{'-' * width}+")
    print(f"         step 1{' ' * (width - 12)}step {step}")

    # Generate final samples
    print(f"\n{'='*60}")
    print(f"FINAL SAMPLES")
    print(f"{'='*60}")
    for temp in [0.5, 0.8, 1.2]:
        print(f"\n  Temperature {temp}:")
        for i in range(3):
            sample = generate(model, tokenizer, prompt="", max_len=100, temperature=temp, block_size=block_size)
            print(f"    {i+1}. \"{sample}\"")

    # Save checkpoint
    ckpt_path = "checkpoint.pt"
    ckpt_data = {
        "model_state_dict": model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
        "model_config": mc,
        "tokenizer_type": mc.tokenizer_type,
        "step": step,
        "train_loss": train_losses[-1],
    }
    if mc.tokenizer_type == "bpe":
        ckpt_data["tokenizer_path"] = mc.tokenizer_path
    else:
        ckpt_data["tokenizer_tokens"] = tokenizer.tokens
    torch.save(ckpt_data, ckpt_path)
    print(f"\n  Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
