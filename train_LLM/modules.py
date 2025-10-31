"""
CSIT5210 - Data Mining and Knowledge Discovery
@author: LI, Weijie | HUANG, Yanzhen
@date: Oct. 10, 2025
@description: Modules for LLM training infrastructure.

@reference:

https://github.com/HappyPointer/LLM2Rec/blob/main/llm2rec/dataset.py

(See full citation in README)

Mainly, this file contains the Tokenizer and the DatasetSuite.

The Tokenizer class originates from the referred repo, but we
made a small patch to make it more memory-friendly.

The DatasetSuite class originates from the referred repo's
PurePromptDataset class. We made a huge upgrade (actually refactor) 
on this class to increase readability and re-usability. 
See the docstring for knowing more about its usage.
"""

import os
import random
import json
import ast

from typing import cast, List, Tuple, Dict, Union
from abc import ABC, abstractmethod
from loguru._logger import Logger

import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase, TrainerCallback
from datasets import Dataset as HFDataset



class Tokenizer:
    """
    A wrapper over a tokenizer instantiated 
    from a pretrained model.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id  #type: ignore
        self.eos_id: int = self.tokenizer.eos_token_id  #type: ignore

    def encode(self, string: str, bos: bool, eos: bool) -> List[int]:
        """
        Using the pre-trained model, encode
        (embed) a string into an embedding.
        
        Parameters:
            s (str): The string to encode.
            bos (bool): Whether to add a beginning-of-sentence token.
            eos (bool): Whether to add an end-of-sentence token.

        Returns:
            List[int]: The encoded embedding.
        """

        assert isinstance(string, str)

        token = self.tokenizer.encode(string) #type: ignore

        # Use a more memory-friendly method.
        # Avoid multiple pandas df object creation.

        ps = 0
        pr = len(token) - 1

        while token[ps] == self.bos_id:
            ps += 1

        while token[pr] == self.eos_id:
            pr -= 1

        token = token[ps:pr+1]

        # Add bos and eos if needed.
        if bos and self.bos_id is not None:
            token = [self.bos_id] + token

        if eos and self.eos_id is not None:
            token = token + [self.eos_id]

        return token
    
    def decode(self, t: List[int]) -> str:
        """
        Decode an embedding into a string
        using the pre-trained model.

        Parameters:
            t (List[int]): The embedding to decode.
        
        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(t) #type: ignore


class DatasetSuite(TorchDataset):
    """
    A dataset suite that works as a middleware between
    the local, grained .csv data and the hugging-face
    formatted dataset.

    Parameters:
        csv_file (str): 
            The path to the grained .csv file.
        tokenizer (AutoTokenizer): 
            The tokenizer.
        max_len (int): 
            Maximum input length.
        sample (int): 
            The number of samples to use. If no need to
            sample, set it to -1, or other negative num.
        is_test (bool): 
            Whether this dataset is used for test,
            rather than train/valid.
        seed (int): 
            The seed for random state. Mainly for 
            reproducibility.
    """
    
    def __init__(self, csv_file: str, 
                 tokenizer: PreTrainedTokenizerBase, 
                 max_len=2048, sample=-1, 
                 is_test = False, seed=0):
        random.seed(seed)
        
        self.data = pd.read_csv(csv_file)

        if not is_test and sample > 0:
            self.data = self.data.sample(
                sample, random_state=seed)

        self.tokenizer = Tokenizer(tokenizer)
        self.test = is_test
        self.max_len = max_len

        # Upon construction, immediately
        # encode the data.
        self.encoded_data = [
            self.__encode_xy(self.__get_xy(row)) 
            for _, row in tqdm(self.data.iterrows())]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]
    
    def __get_xy(self, row) -> Tuple[str, str]:
        """
        Obtained a list of data -> ground-truth pairs
        for supervised learning.

        Parameters:
            row (pd.DataFrame): A row of data.

        Returns:
            Tuple[str, str]: A list of data -> ground-truth pairs.
        """

        # json.loads will fail here with '
        history_item_titles: List[str] = ast.literal_eval(
            row['history_item_titles']) 

        assert isinstance(history_item_titles, list)
        
        history_string = ','.join(history_item_titles)      # Input data

        new_item_title = f'{str(row['new_item_title'])}\n'  # Prediction target

        return (history_string, new_item_title)
    

    def __encode_xy(self, xy: Tuple[str, str]) -> Dict[str, Union[str, List[int]]]:
        """
        Encode a pair of data and ground-truth into
        embeddings. For tests datasets, only the input 
        is encoded. For train datasets, both the input
        and ground-truth are encoded. 
        
        (The output format confronts huggingface.)

        Parameters:
            xy (Tuple[str, str]): 
                A pair of data and ground-truth.

        Returns:
            Dict[str, Union[str, List[int]]]: 
                The encoded pair of data and ground-truth.
        """
        
        history_string, new_item_title = xy

        t_history_string = self.tokenizer.encode(
            history_string, bos=False, eos=False)

        if self.test:
            return {
                'input_ids': t_history_string,
                'attention_mask': [1] * len(t_history_string),
                'text': history_string
            }

        t_new_item_title = self.tokenizer.encode(
            new_item_title, bos=False, eos=True)
        
        tokens = t_history_string + t_new_item_title    # Concatenated Tokens

        labels = (
            [-100] * len(t_history_string) +    # -100 is LLM protocol for label mask
            t_new_item_title)
        
        return {
            'input_ids': tokens[-self.max_len:],    # Last max_len tokens
            'attention_mask': [1] * len(tokens[-self.max_len:]),
            'labels': labels[-self.max_len:]
        }

    
    def to_hf(self) -> HFDataset:
        """
        Convert the encoded data into HuggingFace
        format. Basically, it just "transpose" the
        original dict list (see example).

        (An alternative is to use pandas dataframe,
        but it may require some json parsing, so
        just don't do this.)

        Examples:
        >>> encoded_data = [
        >>>     {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]},
        >>>     {'input_ids': [4, 5, 6], 'attention_mask': [1, 1, 1]}
        >>> ]
        >>> hf_data = encoded_data.to_hf_dict()
        >>> {
        >>>     'input_ids': [[1, 2, 3], [4, 5, 6]],
        >>>     'attention_mask': [[1, 1, 1], [1, 1, 1]]
        >>> }

        Returns:
            HFDataset: The HuggingFace dataset object.
        """

        return HFDataset.from_dict({
            k: [v[k] for v in self.encoded_data]
            for k in self.encoded_data[0].keys()})
    

class TrainSuite(ABC):
    """
    The abstract base class for all training suites.
    All training suites should inherit from this class
    and implement the abstract methods.
    """

    def __init__(self, _logger):
        self.logger = cast(Logger, _logger)

    def _load_config(self, path: str) -> dict:
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

    def _check_exist(self, path: str, what: str, how: str):
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
        
        self.logger.error(
            f'[CSIT5210 Error]: \n\n{what} does not exist! '
            f'Missing {path}. '
            f'Did you {how}?\n\n')
        raise FileNotFoundError(
            f'Base model {path} does not exist.')

    @abstractmethod
    def _get_model_copy(self):
        """
        The method to get a copy of the model.
        """

    @abstractmethod
    def train(self):
        """
        The method to train the model.
        """

    @abstractmethod
    def save(self):
        """
        The method to save the model.
        """


class StopTrainAfterStep(TrainerCallback):
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