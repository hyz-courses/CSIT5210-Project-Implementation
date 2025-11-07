"""
CSIT5210 - Data Mining and Knowledge Discovery
@author: HUANG, Yanzhen | Deng Zhenxiao
@date: Nov 6, 2025
@description: Training and model configurations for downstream model.

Parameters adhere to the original implementation.

@reference:

https://github.com/HappyPointer/LLM2Rec/blob/main

(See full citation in README)
"""

from typing import List
from dataclasses import dataclass, field


@dataclass
class DownstreamTrainArgs:
    """
    Argument configurations for sequential
    recommendation task running.
    """
    num_proc: int = 1
    cache_dir: str = "run_LLM/cache/"
    ckpt_dir: str = "output/downstream/ckpt/"
    rand_seed: int = 2024
    use_pretrained_embedding: bool = True
    
    max_seq_length: int = 10
    whiten: bool = False

    train_batch_size: int = 256
    eval_batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 10_000
    epochs: int = 500

    max_grad_norm: float = 1.0
    eval_interval: int = 5
    patience: int = 20

    topk: List[int] = field(default_factory=lambda: [5, 10, 20])
    run_id: str = "CSIT5210-Implementation-G1"

    save: bool = True

    # Unsettled
    item_num: int = -1
    select_pool: List[int] = field(default_factory=lambda: [-1, -1])
    ext_token_num: int = 0 # extend token number
    eos_token: int = -1


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
    adapter_dims: List[int] = field(default_factory=lambda: [-1])


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
