"""
CSIT5210 - Data Mining and Knowledge Discovery
@author: LI, Weijie | HUANG, Yanzhen
@date: Oct. 10, 2025
@description: Collaborative Supervised Fine-Tuning (CSFT) training suite.
"""

import os
import copy

from loguru import logger

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,

    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import fire

from train_LLM.modules import DatasetSuite
from train_LLM.modules import TrainSuite
from utils.logs import bind_logger

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = bind_logger(logger,
                     log_path=os.path.join(
                        THIS_FILE_DIR, "..",
                        "logs", "csft_train.log"
                     ))


class CSFTTrainSuite(TrainSuite):
    """
    An inheritance of the TrainSuite class for CSFT.
    """

    def __init__(self, trainarg_path: str, steparg_path: str):
        super().__init__(logger)

        # Load configs
        logger.info(
            'Loading configs...\n'
            f'- train args: {trainarg_path}\n'
            f'- step args: {steparg_path}\n\n')

        self.trainarg = self._load_config(trainarg_path)
        self.steparg = self._load_config(steparg_path)

        logger.info('Done!')

        # Prior check if anything is missing
        _base_model_path = self.trainarg['base_model_path']
        _train_data_path = self.trainarg['train_data_path']
        _valid_data_path = self.trainarg['eval_data_path']

        logger.info(
            'Checking missing files...\n'
            f'- Base model: {_base_model_path}\n'
            f'- Train data: {_train_data_path}\n'
            f'- Validation data: {_valid_data_path}')

        self._check_exist(
            _base_model_path, 
            what="Base model", 
            how="download base model")
        
        self._check_exist(
            _train_data_path, 
            what="Train data", 
            how="run data_process.py")
        
        self._check_exist(
            _valid_data_path, 
            what="Validation data", 
            how="run data_process.py")
        
        logger.info('Done! No missing files.')
        
        # Model
        logger.info(
            'Initializing model '
            f'from pretrained {self.trainarg["base_model_path"]}...')

        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.trainarg['base_model_path'],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.finetuned_model = None

        logger.info('Done!')

        # Tokenizer
        logger.info(
            'nitializing tokenizer '
            f'from pretrained {self.trainarg["base_model_path"]}...')

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.trainarg['base_model_path'],
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        logger.info('Done!')

        # Dataset
        logger.info(
            'Initializing dataset suites. '
            'This may take a while...')

        logger.info('Loading training data suite...')

        self.train_data = DatasetSuite(
            csv_file=str(self.trainarg["train_data_path"]),
            tokenizer=self.tokenizer,
            max_len=int(self.trainarg["max_len"]),
            sample=-1,
        ).to_hf()

        logger.info('Done!')

        logger.info('Loading validation data suite...')

        self.val_data = DatasetSuite(
            csv_file=str(self.trainarg["eval_data_path"]),
            tokenizer=self.tokenizer,
            max_len=int(self.trainarg["max_len"]),
            sample=2000,
        ).to_hf()

        logger.info('Done!')

        # Summing up steparg
        logger.info('Summing up steparg...')
        
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

        logger.info('Done!')

    def _get_model_copy(self):
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

        logger.info("Initializing trainer...")

        model_copy = self._get_model_copy()

        logger.info(
            'Copyied a new model instance. \n',
            f'Pre-trained: {id(self.pretrained_model)}, '
            f'Fine-tuned: {id(self.finetuned_model)}'
        )

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

        logger.info("Trainer initialized successfully!")

        logger.info('Fine-tuning model...\n\n')

        model_copy.config.use_cache = False
        trainer.evaluate()
        trainer.train(resume_from_checkpoint=None)

        logger.info('Fine-tuning done! '
                    'Overwriting existing fine-tuned model...')

        self.finetuned_model = model_copy

        logger.info('Done!')
        
    def save(self):
        """
        Save the fine-tuned model to the output directory.
        """

        logger.info(
            'Saving the fine-tuned model...'
        )

        if not self.finetuned_model:
            message = (
                'Model is not fine tuned!'
                'Please train it first!'
            )
            logger.error(message)
            raise ValueError(message)
        
        output_dir = self.trainarg['output_dir']
        if not os.path.exists(output_dir):
            logger.info(
                f'Output directory {output_dir} does not exist. Creating...')

            os.makedirs(output_dir)
        
        self.finetuned_model.save_pretrained(output_dir)

        logger.info(f'Fintuned model saved to {output_dir}!\n\n')


def main():
    config_dir = os.path.join(THIS_FILE_DIR, '..', 'configs')
        
    trainarg, steparg = [
        os.path.join(config_dir, file) 
        for file in ['csft_trainargs.json', 'csft_stepargs.json']]

    csft_train_suite = CSFTTrainSuite(
        trainarg_path=trainarg,
        steparg_path=steparg,
    )

    csft_train_suite.train()
    csft_train_suite.save()

if __name__ == "__main__":
    fire.Fire(main)

