"""Model and training hyperparameters for NeuralNetZero 1M param LLM."""

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
