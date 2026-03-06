# Current Step: Phase 1 -- BPE Tokenizer (8192 vocab)

**Status: IN PROGRESS**

## Why First

Character-level burns ~60% of model capacity learning to spell. "electron" = 8 tokens vs 1-2 with BPE. This is the single biggest bottleneck. Everything downstream depends on a good tokenizer.

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `tokenizer.py` | Modified | Added `BPETokenizer` class alongside existing `CharTokenizer` |
| `config.py` | Modified | Added `tokenizer_type`, `tokenizer_path` fields + `CogCore500MConfig` |
| `train.py` | Modified | Tokenizer selection by config, checkpoint metadata update |
| `inference.py` | Modified | BPE tokenizer loading from checkpoint (backward-compatible) |
| `prepare_corpus.py` | New | Build tokenizer training corpus from data + synthetic STEM text |
| `train_tokenizer.py` | New | Train BPE tokenizer using HuggingFace `tokenizers` |
| `tokenizer/` | New dir | Output: `stem_bpe.json` |

## Sub-tasks

- [ ] `pip install tokenizers`
- [ ] `python prepare_corpus.py` -> `tokenizer/corpus.txt`
- [ ] `python train_tokenizer.py` -> `tokenizer/stem_bpe.json`
- [ ] `BPETokenizer` class in `tokenizer.py`
- [ ] Config updates in `config.py`
- [ ] `train.py` tokenizer selection
- [ ] `inference.py` backward-compatible loading
- [ ] Validation: roundtrip, compression ratio, STEM terms, digit splitting

## Acceptance Criteria

1. `tokenizer/stem_bpe.json` exists with 8192 vocab
2. `BPETokenizer` class works identically to `CharTokenizer` interface
3. `python train.py` works with both char and BPE tokenizers
4. `python inference.py` loads BPE checkpoints correctly
5. Old char-tokenizer checkpoints still work (backward compat)
6. STEM terms compress to 1-2 tokens
7. Compression ratio >= 3x vs char tokenizer

## What Comes Next

Phase 2: Data Pipeline -- ClimbMix clustering, synthetic generation scripts, document packing.
