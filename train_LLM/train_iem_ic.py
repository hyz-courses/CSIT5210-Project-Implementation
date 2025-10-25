import os

from loguru import logger

from transformers import TrainingArguments, set_seed

from accelerate import Accelerator

from train_LLM.data_classes import DataArgs, ModelArgs
from train_LLM.modules import TrainSuite

from utils.logs import bind_logger

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = bind_logger(logger,
                     log_path=os.path.join(
                        THIS_FILE_DIR, "..",
                        "logs", "iem_ic.log"
                     ))


class ICTrainSuite(TrainSuite):
    
    def __init__(self, config_path: str):
        super().__init__(logger)

        args = [
            f"ic_{usage}_args.json"
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

        self.accelerator = Accelerator()

        set_seed(self.train_args.seed)

        if self.train_args.gradient_checkpointing:
            self.train_args.gradient_checkpointing_kwargs = {"use_reentrant": False}


