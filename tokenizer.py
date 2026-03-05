"""Character-level tokenizer for NeuralNetZero."""


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
