"""
CSIT5210 - Data Mining and Knowledge Discovery
@author: YE, Huaxiang | HUANG, Yanzhen
@date: Oct. 31, 2025
@description: Train Suite for IEM stage 2: 
SimCSE (Simple Contrastive Learning of Sentence Embeddings) training.

@reference:

https://github.com/HappyPointer/LLM2Rec/blob/main/llm2rec/dataset.py

@misc{gao2022simcsesimplecontrastivelearning,
      title={SimCSE: Simple Contrastive Learning of Sentence Embeddings}, 
      author={Tianyu Gao and Xingcheng Yao and Danqi Chen},
      year={2022},
      eprint={2104.08821},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2104.08821}, 
}

(See full citation in README)
"""

import os
import random
from copy import copy
from typing import List, Dict, Union, Any, Optional, cast


from loguru import logger

import torch
from torch import Tensor
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from transformers import TrainingArguments, set_seed, Trainer
from llm2vec import LLM2Vec
from accelerate import Accelerator

from train_LLM.data_classes import (
    DataArgs, ModelArgs, 
    DataSample, SentenceExample)
from train_LLM.modules import TrainSuite, StopTrainAfterStep
from utils.logs import bind_logger

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.join(THIS_FILE_DIR, "..")

logger = bind_logger(logger,
                     log_path=os.path.join(
                        THIS_FILE_DIR, "..",
                        "logs", "iem_ic.log"
                     ))


class ItemTitleDataset(Dataset):
    """
    Dataset of lines of item titles.
    """
    
    def __init__(
            self, file_path: str,
            separator: str = "[SEP]"):
        
        # Read all raw titles
        self.raw_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.raw_data.append(line.strip())
            f.close()

        # Generate data samples
        _data_samples = [
            DataSample(
                id_=i,
                query= separator + line,
                positive= separator + line,
                task_name="AmazonMix")
            for i, line in enumerate(self.raw_data)
        ]

        random.shuffle(_data_samples)
        self.data_samples = _data_samples
    
    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        sample: DataSample = self.data_samples[index]
        return SentenceExample(
            texts=[sample.query, sample.positive],
            label=1.0
        )
    
    def __iter__(self):
        return iter(self.data_samples)
    

class InfoNCEHN:
    """
    InfoNCE loss with hard negatives for contrasive learning.
    Modified from source code for single-gpu.
    """
    
    def __init__(self):
        pass

    def _cos_sim(self, n1: Tensor, n2: Tensor) -> Tensor:
        assert isinstance(n1, Tensor) and isinstance(n2, Tensor)

        n1_, n2_ = [normalize(x) for x in [n1, n2]]

        return torch.mm(n1_, n2_.transpose(0, 1))

    def __call__(
            self, 
            query: Tensor, 
            docreps_pos: Tensor, 
            docreps_neg: Tensor):
    
        # Only one GPU

        docreps = torch.cat([docreps_pos, docreps_neg], dim=0)
        scores = self._cos_sim(query, docreps) * 20.0
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)

        return torch.nn.CrossEntropyLoss()(scores, labels)
    

class SentenceCollator:
    """
    Sentence collator for SimCSE.
    """
    
    def __init__(self, model: LLM2Vec):
        self.model = model
    
    def __call__(self, features: List[SentenceExample]):
        """
        >>> @dataclass(init=True)
        >>> class SentenceExample:
        >>>     guid: str = ""
        >>>     label: Union[int, float] = 0
        >>>     texts: List[str]

        Batch tokenize text list for all sentence examples.
        """

        num_texts = len(features[0].texts)
        texts = [[]] * num_texts
        labels = []

        for example in features:
            for i, text in enumerate(example.texts):
                texts[i].append(text)
            labels.append(example.label)

        sentence_features = [
            self.model.tokenize(texts[i])
            for i in range(num_texts)
        ]

        return sentence_features, torch.tensor(labels)
        


class SimCSETrainer(Trainer):
    """
    An inheritance of the Trainer class for SimCSE.
    (Inherited from paper's source code.)
    """

    def _save(self, output_dir: Optional[str] = None, state_dict=None): # pylint: disable=unused-argument
        
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        output_dir = os.path.join(PROJECT_ROOT_DIR, output_dir)
        
        os.makedirs(output_dir, exist_ok=True)

        self.model = cast(LLM2Vec, self.model)
        self.model.save(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
    
    def compute_loss(
            self, model: torch.nn.Module, 
            inputs: Dict[str, Union[Tensor, Any]],
            return_outputs: bool = False):
        
        features, _ = inputs # unsupervised
        query, doc_pos = [self.model(x) for x in features[:2]]
        doc_neg = None if len(features) <= 2 else self.model(features[2])

        loss = InfoNCEHN()(query, doc_pos, doc_neg)
        
        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] 
                 for row in features], dim=1
            )
            return loss, output

        return loss


class SimCSETrainSuite(TrainSuite):
    """
    An inheritance of the TrainSuite class for 
    unsupervised SimCSE training.
    """
    
    def __init__(self, config_path: str):
        super().__init__(logger)

        # Load Arguments from Config
        args = [
            f"simcse_{usage}_args.json"
            for usage in [
                "model", "datatraining", "training"]]
        arg_paths = [os.path.join(config_path, arg) for arg in args]

        for arg_path in arg_paths:
            self._check_exist(
                arg_path, "configuration json", 
                "delete the configuration json files")
        
        _model_args, _data_args, _train_args = [
            self._load_config(arg_path) for arg_path in arg_paths]
        
        self.train_args = TrainingArguments(**_train_args)
        self.data_args = DataArgs(**_data_args)
        self.model_args = ModelArgs(**_model_args)

        # Training
        set_seed(self.train_args.seed)

        self.accelerator = Accelerator()
        if self.train_args.gradient_checkpointing:
            self.train_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        
        # Dataset
        # Note that the Qwen Series do not accept any BERT-like
        # separators like [SEP]. Need to use natural language separators.
        # The original author uses `!@#$%^&*()`, which may cause 
        # some ambiguity.
        self.train_dataset = ItemTitleDataset(
            file_path=os.path.join(
                PROJECT_ROOT_DIR, 
                self.data_args.dataset_file_path), 
            separator="ProductName: ")

        # Model
        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=self.model_args.model_name_or_path,
            enable_bidirectional=self.model_args.bidirectional,
            merge_peft=True,
            pooling_mode=self.model_args.pooling_mode,
            max_length=self.model_args.max_seq_length,
            torch_dtype=self.model_args.torch_dtype,
            attn_implementation=self.model_args.attn_implementation,
            attention_dropout=0.2
        )

        for param in self.model.model.parameters():
            param.requires_grad = True

        self.tokenizer = self.model.tokenizer
    
    def train(self):

        trainer = SimCSETrainer(
            model=self.model,
            args=self.train_args,
            train_dataset=list(self.train_dataset),
            data_collator=SentenceCollator(self.model),
            tokenizer=self.tokenizer)
    
        trainer.add_callback(StopTrainAfterStep(self.train_args.max_steps))

        trainer.train()

    def _get_model_copy(self):
        return copy(self.model)

    def save(self):
        return


if __name__ == "__main__":
    config_dir = os.path.join(PROJECT_ROOT_DIR, "configs")
    
    simcse_train_suite = SimCSETrainSuite(config_path=config_dir)
    simcse_train_suite.train()

