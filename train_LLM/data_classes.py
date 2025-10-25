from typing import Optional
from dataclasses import dataclass

@dataclass(init=True)
class ModelArgs:
    model_name_or_path: str
    torch_dtype: str
    attn_implementation: str
    trust_remote_code: bool

    peft_model_name_or_path: Optional[str] = None
    bidirectional: Optional[bool] = None
    max_seq_length: Optional[int] = None
    pooling_mode: Optional[str] = None

    
@dataclass(init=True)
class DataArgs:

    dataset_name: Optional[str] = None
    dataset_file_path: Optional[str] = None
    line_by_line: Optional[bool] = None
    max_seq_length: Optional[int] = None
    mlm_probability: Optional[float] = None

    processing_num_workers: Optional[int] = None
    pad_to_max_length: Optional[bool] = False
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    streaming: Optional[bool] = False

    overwrite_cache: Optional[bool] = True