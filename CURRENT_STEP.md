# Current Step: Phase 1b -- Retrain Tokenizer to 128K Vocab

**Status: NOT STARTED**

**Previous: Phase 1a (BPE Tokenizer) -- COMPLETE** (8192 vocab, 5.0x compression, all tests pass)

## Why This Step

CR5 mandates vocab=128000 (2026 frontier minimum is 65K). The tokenizer is the foundation -- all downstream data processing, model architecture, and embeddings depend on it. This blocks Phase 2.

## Sub-tasks

### Config Updates
- [ ] Update `config.py`: vocab_size = 128000
- [ ] Update `train_tokenizer.py`: vocab_size = 128000

### Corpus Expansion
- [ ] Significantly expand corpus (128K merges need much more data -- target 50-100MB corpus)
- [ ] Include multilingual data: Serbian Cyrillic text, German text with umlauts/ß
- [ ] Include code samples (Python, C++, JavaScript keywords and operators)
- [ ] Include STEM notation (LaTeX, chemical formulas, mathematical symbols)

### Training
- [ ] Run `prepare_corpus.py` (expanded)
- [ ] Run `train_tokenizer.py`

### Verification
- [ ] Verify: vocab=128000, compression >= 15x bytes, digits split, roundtrip clean
- [ ] Verify STEM tokens: \frac, \partial, \int, \sum -> 1 token each
- [ ] Verify trilingual coverage: Serbian Cyrillic, German umlauts/ß -> single tokens
- [ ] Verify code tokens: common keywords, operators -> single tokens

### Architecture Prep
- [ ] Implement embedding projection layer (640->1280)

## Acceptance Criteria

1. Tokenizer trained with vocab=128000
2. Compression ratio >= 15x bytes
3. Digits still split individually
4. Roundtrip encoding/decoding is clean (no data loss)
5. Key STEM tokens (\frac, \partial, \int, \sum) are 1 token each
6. Serbian Cyrillic and German umlauts/ß are single tokens
7. Common code keywords/operators are single tokens
8. All existing tests pass with new tokenizer

## What Comes Next

Phase 2: Data Pipeline -- implement CLIMB-style clustering on FineWeb-Edu, source code/math/science data, generate synthetic reasoning traces, validate domain ratios.