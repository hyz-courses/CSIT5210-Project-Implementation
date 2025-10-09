"""
CSIT5210 - Data Mining and Knowledge Discovery
@author: LI, Weijie | HUANG, Yanzhen
@date: Oct. 10, 2025
@description: Collaborative Supervised Fine-Tuning (CSFT) training suite.
"""

import os
import json
import copy

from loguru import logger

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from train_LLM.modules import DatasetSuite

logger.add("logs/train_csft.log", rotation="10 MB")


class CSFTTrainSuite:
    """
    The trainer class for CSFT.
    """

    def __init__(self, trainarg_path: str, steparg_path: str):

        # Load configs
        logger.info(
            '[CSIT5210 Info]:\n\nLoading configs...\n'
            f'- train args: {trainarg_path}\n'
            f'- step args: {steparg_path}\n\n')

        self.trainarg = self.__load_config(trainarg_path)
        self.steparg = self.__load_config(steparg_path)

        logger.info('[CSIT5210 Info]: Done!')

        # Prior check if anything is missing
        _base_model_path = self.trainarg['base_model_path']
        _train_data_path = self.trainarg['train_data_path']
        _valid_data_path = self.trainarg['eval_data_path']

        logger.info(
            '[CSIT5210 Info]:\n\nChecking missing files...\n'
            f'- Base model: {_base_model_path}\n'
            f'- Train data: {_train_data_path}\n'
            f'- Validation data: {_valid_data_path}\n\n')

        self.__check_exist(
            _base_model_path, 
            what="Base model", 
            how="download base model")
        
        self.__check_exist(
            _train_data_path, 
            what="Train data", 
            how="run data_process.py")
        
        self.__check_exist(
            _valid_data_path, 
            what="Validation data", 
            how="run data_process.py")
        
        logger.info('[CSIT5210 Info]: Done! No missing files.')
        
        # Model
        logger.info(
            '[CSIT5210 Info]:\n\nInitializing model '
            f'from pretrained {self.trainarg["base_model_path"]}...\n\n')

        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.trainarg['base_model_path'],
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.finetuned_model = None

        logger.info('[CSIT5210 Info]: Done!')

        # Tokenizer
        logger.info(
            '[CSIT5210 Info]:\n\nInitializing tokenizer '
            f'from pretrained {self.trainarg["base_model_path"]}...\n\n')

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.trainarg['base_model_path'],
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        logger.info('[CSIT5210 Info]: Done!')

        # Dataset
        logger.info(
            '[CSIT5210 Info]:\n\nInitializing dataset suites. '
            'This may take a while...\n\n')

        logger.info('[CSIT5210 Info]:\n\n'
                    'Loading training data suite...\n\n')

        self.train_data = DatasetSuite(
            csv_file=str(self.trainarg["train_data_path"]),
            tokenizer=self.tokenizer,
            max_len=int(self.trainarg["max_len"]),
            sample=-1,
        ).to_hf()

        logger.info('[CSIT5210 Info]: Done!')

        logger.info('[CSIT5210 Info]:\n\n'
                    'Loading validation data suite...\n\n')

        self.val_data = DatasetSuite(
            csv_file=str(self.trainarg["eval_data_path"]),
            tokenizer=self.tokenizer,
            max_len=int(self.trainarg["max_len"]),
            sample=2000,
        ).to_hf()

        logger.info('[CSIT5210 Info]: Done!')

        # Summing up steparg
        logger.info('[CSIT5210 Info]:\n\n'
                    'Summing up steparg...\n\n')
        
        gradient_accumulation_steps: int = int(self.trainarg["batch_size"]) // int(
            self.trainarg["micro_batch_size"]
        )

        self.steparg.update(
            {
                "per_device_train_batch_size": int(self.trainarg["micro_batch_size"]),
                "per_device_eval_batch_size": int(self.trainarg["micro_batch_size"]),
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_train_epochs": int(self.trainarg["num_epochs"]),
                "learning_rate": float(self.trainarg["learning_rate"]),
                "output_dir": str(self.trainarg["output_dir"]),
            }
        )

        logger.info('[CSIT5210 Info]: Done!')

    def __check_exist(self, path: str, what: str, how: str):
        """
        Check whether a key file is exist.
        Parameters:
            path (str): 
                The path of the key file.
            what (str): 
                The name of the key file.
            how (str): 
                How to get the key file.
        """
        if os.path.exists(path):
            return
        
        logger.error(
            f'[CSIT5210 Error]: \n\n{what} does not exist! '
            f'Missing {path}. '
            f'Did you {how}?\n\n')
        raise FileNotFoundError(
            f'Base model {path} does not exist.')

    def __load_config(self, path: str) -> dict:
        """
        Load configuration from a .json file.
        Parameters:
            config_path (str):
                Path to the .json file.
        Returns:
            dict:
                Configuration dictionary.
        """

        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
            f.close()
        return config

    def __get_model_copy(self):
        """
        Clone the loaded model to a new memory space.
        Returns:
            Model:
                Cloned model.
        """
        return copy.copy(self.pretrained_model)

    def train(self):
        """
        Train the model.
        """

        logger.info("[CSIT5210 Info]: Initializing trainer...")

        model_copy = self.__get_model_copy()

        trainer = Trainer(
            model=model_copy,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            args=TrainingArguments(**self.steparg),
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        logger.info("[CSIT5210 Info]: Trainer initialized successfully!")

        logger.info('[CSIT5210 Info]: \n\nFine-tuning model...\n\n')

        model_copy.config.use_cache = False
        trainer.evaluate()
        trainer.train(resume_from_checkpoint=None)

        logger.info('[CSIT5210 Info]: \n\nFine-tuning done! '
                    'Overwriting existing fine-tuned model...\n\n')

        self.finetuned_model = model_copy

        logger.info('[CSIT5210 Info]: Done!')

        
    def save(self):
        """
        Save the fine-tuned model to the output directory.
        """

        logger.info(
            '[CSIT5210 Info]: \n\nSaving the fine-tuned model...\n\n'
        )

        if not self.finetuned_model:
            message = (
                '[CSIT5210 Error]: \n\nModel is not fine tuned!'
                'Please train it first!\n\n'
            )
            logger.error(message)
            raise ValueError(message)
        
        output_dir = self.trainarg['output_dir']
        if not os.path.exists(output_dir):
            logger.info(
            '[CSIT5210 Info]: \n\n'
            f'Output directory {output_dir} does not exist. '
            'Creating...\n\n')

            os.makedirs(output_dir)
        
        self.finetuned_model.save_pretrained(output_dir)

        logger.info(
            '[CSIT5210 Info]: \n\n '
            f'Fintuned model saved to {output_dir}!\n\n'
        )


if __name__ == "__main__":
    config_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'configs')
    
    trainarg, steparg = [
        os.path.join(config_dir, file) 
        for file in ['csft_trainargs.json', 'csft_stepargs.json']]

    csft_train_suite = CSFTTrainSuite(
        trainarg_path=trainarg,
        steparg_path=steparg,
    )

    csft_train_suite.train()
    csft_train_suite.save()
