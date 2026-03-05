"""Interactive text generation from a trained NeuralNetZero checkpoint."""

import sys
import torch

from model import GPT
from tokenizer import CharTokenizer


def generate(model, tokenizer, prompt: str = "", max_len: int = 200,
             temperature: float = 0.8, top_k: int = 20) -> str:
    model.eval()
    device = next(model.parameters()).device

    if prompt:
        ids = tokenizer.encode(prompt)
    else:
        ids = [tokenizer.bos_id]

    ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            idx_cond = ids[:, -model.config.block_size:]
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

    return tokenizer.decode(ids[0].tolist())


def load_model(ckpt_path: str = "checkpoint.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]

    # Rebuild tokenizer from saved tokens
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.tokens = ckpt["tokenizer_tokens"]
    tokenizer.token_to_id = {t: i for i, t in enumerate(tokenizer.tokens)}
    tokenizer.id_to_token = {i: t for i, t in enumerate(tokenizer.tokens)}
    tokenizer.pad_token = "<|pad|>"
    tokenizer.bos_token = "<|bos|>"
    tokenizer.eos_token = "<|eos|>"
    tokenizer.pad_id = tokenizer.token_to_id[tokenizer.pad_token]
    tokenizer.bos_id = tokenizer.token_to_id[tokenizer.bos_token]
    tokenizer.eos_id = tokenizer.token_to_id[tokenizer.eos_token]
    tokenizer.special_tokens = [tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token]

    model = GPT(mc)
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Loaded model: {model.count_parameters():,} params from {ckpt_path}")
    print(f"  Trained for {ckpt['step']} steps, final loss: {ckpt['train_loss']:.4f}")
    print(f"  Device: {device}")

    return model, tokenizer


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint.pt"
    model, tokenizer = load_model(ckpt_path)

    print("\nInteractive mode. Type a prompt (or empty for unconditional generation).")
    print("Commands: :temp N (set temperature), :topk N (set top-k), :quit\n")

    temperature = 0.8
    top_k = 20

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prompt == ":quit":
            break
        elif prompt.startswith(":temp "):
            temperature = float(prompt.split()[1])
            print(f"  Temperature set to {temperature}")
            continue
        elif prompt.startswith(":topk "):
            top_k = int(prompt.split()[1])
            print(f"  Top-k set to {top_k}")
            continue

        text = generate(model, tokenizer, prompt=prompt, max_len=200,
                        temperature=temperature, top_k=top_k)
        print(f"  {text}\n")


if __name__ == "__main__":
    main()
