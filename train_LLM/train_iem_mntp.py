"""
CSIT5210 - Data Mining and Knowledge Discovery
@author: LI, Yixin | HUANG, Yanzhen
@date: Oct. 22, 2025
@description: Train Suite for IEM stage 1: 
MNTP (Masked Next Token Prediction) training.

@reference:

https://github.com/HappyPointer/LLM2Rec/blob/main/llm2rec/dataset.py

(See full citation in README)

Mainly, this file uses MNTPTrainer and StopTrainingCallback
from the source code to assist our implementation.
"""

import os

from copy import copy
from typing import cast, Union, Type, Tuple
from pathlib import Path

import numpy as np
import torch
import evaluate
from datasets import load_dataset, DatasetDict
from torch import Tensor
from torch.nn import Embedding
from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback)
from llm2vec.models import (
    MistralBiForMNTP,
    LlamaBiForMNTP,
    GemmaBiForMNTP,
    Qwen2BiForMNTP,
)

from train_LLM.modules import TrainSuite
from train_LLM.data_classes import DataArgs, ModelArgs

UnionBiForMNTP = Union[ # pylint: disable=invalid-name
    MistralBiForMNTP,
    LlamaBiForMNTP,
    GemmaBiForMNTP,
    Qwen2BiForMNTP,
]

class MNTPTrainer(Trainer):
    """
    An inheritance of the Trainer class for MNTP.
    (Inherited from paper's source code.)
    """
    
    def __init__(self, *args, **kwargs):
        self.label_names = ["labels"]
        super().__init__(*args, **kwargs)
    
    def _save(self):
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Ensure that the model is of the correct type
        # s.t. the inner model could be saved by save_peft_model.
        self.model = cast(UnionBiForMNTP, self.model)
        self.tokenizer = cast(PreTrainedTokenizerBase, self.tokenizer)

        self.model.save_peft_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Train arguments saving for re-productability.
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


class StopTrainCallback(TrainerCallback):
    """
    The callback class for stopping training
    after a certain #. of steps.
    (Inherited from paper's source code.)
    """
    
    def __init__(self, after_step: int):
        self.after_step = after_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.after_step:
            control.should_training_stop = True


class MNTPTrainSuite(TrainSuite):
    """
    An inheritance of the TrainSuite class for MNTP.
    """
    
    def __init__(self, config_path: str):
        super().__init__()

        # ===== 0. Preflight ===== 
        # Check if there's any missing essential files.
        
        # 0.1 Configuration json files
        args = [
            f"mntp_{usage}_args.json" 
            for usage in [
                "model", "datatraining", "training"]]
        arg_paths = [os.path.join(config_path, arg) for arg in args]

        for arg_path in arg_paths:
            self._check_exist(
                arg_path, "configuration json", 
                "delete the configuration json files")
        
        # Load configurations
        _model_args, _data_args, _train_args = [
            self._load_config(arg_path) for arg_path in arg_paths
        ]

        self.train_args = TrainingArguments(**_train_args)
        self.data_args = DataArgs(**_data_args)
        self.model_args = ModelArgs(**_model_args)

        # 0.2 Dataset files
        dataset_path = self.data_args.dataset_name
        self._check_exist(dataset_path, "MNTP dataset", "run data_process.py")

        set_seed(self.train_args.seed)

        # cache_dir: None, revision: main, token: None, trust: false, use_fast: true

        # ===== 1. Configs =====
        self.config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=self.model_args.trust_remote_code
        )

        # ===== 2. Tokenizer =====
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=self.model_args.trust_remote_code,
            use_fast=True
        )

        if self.tokenizer.mask_token is None:
            self.tokenizer.mask_token = "_"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ===== 3. Model =====
        model_class: Type[UnionBiForMNTP] = {
            "MistralConfig": MistralBiForMNTP,
            "LlamaConfig": LlamaBiForMNTP, 
            "GemmaConfig": GemmaBiForMNTP,
            "Qwen2Config": Qwen2BiForMNTP,
        }[self.config.__class__.__name__]

        model_name_or_path = self.model_args.model_name_or_path

        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf = Path(model_name_or_path).suffix == ".ckpt",
            config = self.config,
            trust_remote_code = self.model_args.trust_remote_code,
            torch_dtype = self.model_args.torch_dtype,
            low_cpu_mem_usage = False,
            attn_implementation = self.model_args.attn_implementation,
        )

        model = cast(UnionBiForMNTP, model)

        for param in model.model.parameters():
            param.requires_grad = True

        embedding = model.get_input_embeddings()

        assert isinstance(embedding, Embedding)

        if len(self.tokenizer) > embedding.weight.shape[0]:
            model.resize_token_embeddings(len(self.tokenizer))

        self.model = model

        # ===== 4. Load and tokenize datasets =====

        raw_datasets = load_dataset("text", data_files=dataset_path)
        assert isinstance(raw_datasets, DatasetDict)

        split_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
        raw_datasets['train'] = split_datasets["train"]
        raw_datasets['train'] = raw_datasets['train'].shuffle(seed=self.train_args.seed)
        raw_datasets['validation'] = split_datasets["test"]
        
        with self.train_args.main_process_first():
            tokenized_datasets = raw_datasets.map(
                lambda examples: self.tokenizer(
                    [
                        line for line in examples["text"]
                        if len(line) > 0 and not line.isspce()
                    ],
                    padding= (
                        "max_length" 
                        if self.data_args.pad_to_max_length
                        else False),
                    truncation=True,
                    max_length=min(
                        self.data_args.max_seq_length, 
                        self.tokenizer.model_max_length),
                    return_special_tokens_mask=True,
                ),
                batched=True,
                num_proc=self.data_args.processing_num_workers,
                remove_columns=["text"],
                load_from_cache_file = not self.data_args.overwrite_cache,
            )

        self.train_dataset = tokenized_datasets["train"]
        self.eval_dataset = tokenized_datasets["validation"]

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.data_args.mlm_probability,
            pad_to_multiple_of=8 if self.train_args.fp16 else None
        )

    def train(self):
        
        metric_acc = evaluate.load("accuracy")

        def preprocess_logits_for_metrics(
                logits: Union[
                    Tensor, 
                    Tuple[Tensor, ...]]) -> Tensor:
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)
        
        def comopute_metrics(eval_preds: Tuple[Tensor, Tensor]):
            preds, labels = eval_preds
            preds, labels = [
                preds[:, :-1].reshape(-1), 
                labels[:, 1:].reshape(-1)]
            mask = labels != -100
            preds, labels = preds[mask], labels[mask]
            return metric_acc.compute(predictions=preds, references=labels)

        trainer = MNTPTrainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=comopute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        trainer.add_callback(
            StopTrainCallback(after_step=1000))
        
        # ==== Training ====
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # ==== Evaluation ====
        metrics = trainer.evaluate()

        # Inv prob.
        perplexity = np.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    def _get_model_copy(self):
        return copy(self.model)

    def save(self):
        return


if __name__ == "__main__":
    config_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "configs"
    )

    mntp_train_suite = MNTPTrainSuite(config_path=config_dir)
    mntp_train_suite.train()