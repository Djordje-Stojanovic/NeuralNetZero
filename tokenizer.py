"""Tokenizers for NeuralNetZero: character-level and BPE."""


class CharTokenizer:
    def __init__(self, texts: list[str]):
        # Special tokens
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"
        self.pad_token = "<|pad|>"

        # Build vocab from data
        chars = sorted(set("".join(texts)))
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token]
        self.tokens = self.special_tokens + chars

        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for i, t in enumerate(self.tokens)}

        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[ch] for ch in text if ch in self.token_to_id]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            tok = self.id_to_token.get(i, "")
            if tok not in self.special_tokens:
                out.append(tok)
        return "".join(out)


class BPETokenizer:
    """BPE tokenizer wrapping HuggingFace tokenizers library."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(path)
        self.pad_id = self._tokenizer.token_to_id("<|pad|>")
        self.bos_id = self._tokenizer.token_to_id("<|bos|>")
        self.eos_id = self._tokenizer.token_to_id("<|eos|>")

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)
