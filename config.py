"""Model and training hyperparameters for NeuralNetZero."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_layer: int = 5
    d_model: int = 128
    n_head: int = 4
    d_head: int = 32          # d_model // n_head
    ffn_inner: int = 344      # ~2.67x d_model for SwiGLU
    block_size: int = 512
    vocab_size: int = -1      # set from tokenizer
    dropout: float = 0.0
    tokenizer_type: str = "char"    # "char" or "bpe"
    tokenizer_path: str = ""        # path to BPE JSON (ignored for char)


@dataclass
class CogCore500MConfig(ModelConfig):
    n_layer: int = 32
    d_model: int = 896
    n_head: int = 14
    d_head: int = 64
    ffn_inner: int = 448
    block_size: int = 2048
    vocab_size: int = 8192
    tokenizer_type: str = "bpe"
    tokenizer_path: str = "tokenizer/stem_bpe.json"


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    muon_lr: float = 0.02
    weight_decay: float = 0.1
    muon_weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    grad_clip: float = 1.0
    eval_interval: int = 100
    sample_interval: int = 200
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
