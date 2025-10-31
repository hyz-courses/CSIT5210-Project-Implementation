"""
CSIT5210 - Data Mining and Knowledge Discovery
@author: HUANG, Yanzhen
@date: Oct. 22, 2025
@description: Dataclasses for CSFT and IEM.

@reference:

https://github.com/HappyPointer/LLM2Rec/blob/main/llm2rec/dataset.py

(See full citation in README)
"""


from typing import Optional, List, Union
from dataclasses import dataclass


@dataclass(init=True)
class ModelArgs:
    """
    Runtime arguments for model class initialization
    and weights loading.
    """
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
    """
    Runtime arguments for dataset processing.
    """

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


@dataclass(init=True)
class DataSample:
    """
    Data sample for contrastive learning.
    Contains a query, a positive sample, and an optional negative sample.
    """
    id_: int
    query: str
    positive: str
    negative: Optional[str] = None
    task_name: Optional[str] = None
    aug_query: Optional[str] = None


@dataclass(init=True)
class SentenceExample:
    """
    Sentence example for contrastive learning.
    Contains a query and a positive sample.
    """
    texts: List[str]
    guid: str = ""
    label: Union[int, float] = 0
    
    