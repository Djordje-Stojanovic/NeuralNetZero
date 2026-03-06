"""Train BPE tokenizer using HuggingFace tokenizers library."""

import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, ByteLevel, Digits
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def main():
    corpus_path = os.path.join("tokenizer", "corpus.txt")
    output_path = os.path.join("tokenizer", "stem_bpe.json")

    if not os.path.exists(corpus_path):
        print(f"ERROR: {corpus_path} not found. Run prepare_corpus.py first.")
        return

    print("Training BPE tokenizer...")
    print(f"  Corpus: {corpus_path}")

    # Configure tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Sequence([
        ByteLevel(add_prefix_space=False),
        Digits(individual_digits=True),
    ])
    tokenizer.decoder = ByteLevelDecoder()

    # Special tokens (order determines IDs: PAD=0, UNK=1, BOS=2, EOS=3, ...)
    special_tokens = [
        "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
        "<think>", "</think>", "<|user|>", "<|assistant|>",
    ]

    trainer = BpeTrainer(
        vocab_size=8192,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    # Train
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(output_path)

    print(f"\n  Saved to {output_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Verify special token IDs
    print("\n  Special tokens:")
    for tok in special_tokens:
        tid = tokenizer.token_to_id(tok)
        print(f"    {tok} -> {tid}")

    # Sample tokenizations
    print("\n  Sample tokenizations:")
    samples = [
        "acceleration",
        "photosynthesis",
        "eigenvalue",
        "F=ma",
        "E=mc^2",
        "H2O",
        "12345",
        "the derivative of x^n is n*x^(n-1)",
        "force equals mass times acceleration",
    ]
    for text in samples:
        encoding = tokenizer.encode(text)
        ids = encoding.ids
        decoded = tokenizer.decode(ids)
        # Use repr for tokens to avoid Windows console encoding issues
        tokens_str = [t.encode("utf-8", errors="replace").decode("ascii", errors="replace") for t in encoding.tokens]
        print(f"    '{text}' -> {len(ids)} tokens: {tokens_str}")
        if decoded != text:
            print(f"      WARNING: roundtrip mismatch: '{decoded}'")


if __name__ == "__main__":
    main()
