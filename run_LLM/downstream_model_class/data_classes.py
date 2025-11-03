from typing import List
from dataclasses import dataclass


@dataclass
class SeqRecArgs:
    """
    Argument configurations for sequential
    recommendation task running.
    """
    max_seq_length: int = 10
    whiten: bool = False

    train_batch_size: int = 256
    eval_batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 10_000
    epochs: int = 1000

    max_grad_norm: float = 1.0
    eval_interval: int = 5
    patience: int = 20

    topk: List[int] = [5, 10, 20]
    run_id: str = "CSIT5210-Implementation-G1"

    save: bool = True

    # Unsettled
    item_num: int = -1
    ext_token_num: int = 0 # extend token number


@dataclass
class DownstreamModelArgs:
    """
    Common model arguments for downstream
    sequential recommendation models.
    """
    loss_type: str = "ce"
    hidden_size: int = 128
    layer_num: int = 2
    dropout: float = 0.3
    sample_func: str = "random"
    adapter_dims: List[int] = [-1]


@dataclass
class SASRecModelArgs(DownstreamModelArgs):
    """
    Parameter specification for SASRec model.
    """
    num_heads: int = 2


@dataclass
class GRU4RecModelArgs(DownstreamModelArgs):
    """
    Parameter specification for GRU4Rec model.
    """
    lr: float = 1e-2
