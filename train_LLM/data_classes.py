from typing import Optional
from dataclasses import dataclass

@dataclass(init=True)
class ModelArgs:
    model_name_or_path: str
    torch_dtype: str
    attn_implementation: str
    trust_remote_code: bool

    
@dataclass(init=True)
class DataArgs:

    dataset_name: str
    line_by_line: bool
    max_seq_length: int
    mlm_probability: float

    processing_num_workers: Optional[int] = None
    pad_to_max_length: bool = False
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    streaming: bool = False

    overwrite_cache: bool = True