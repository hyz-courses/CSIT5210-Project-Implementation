import os
import ast

from typing import cast, List

import torch
from torch.utils.data import Dataset

from data_process import CSVLoader


class IDRecDataset(Dataset):
    """
    Traditional sequential recommendation dataset
    where each item is represented by an ID.
    """

    def __init__(self, max_len: int, category: str, usage: str):
        self.category = category
        self.max_len = max_len
        self.usage = usage
        self.raw_data = self.load_data()
    
    def load_data(self):
        """
        Load a csv data from the category and 
        extract the two columns of `history_item_ids`
        and `new_item_id`. Put into a single sequence
        as raw data.
        """

        project_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".."
        )
        
        df = CSVLoader(
            category=self.category,
            phase="grained",
            usage=self.usage,
            limit=172747,
            project_root=project_root
        ).load()

        history_item_ids_list = df["history_item_ids"]
        new_item_id_list = df["new_item_id"]

        sequences = [
            cast(List, ast.literal_eval(history_item_ids)) + [new_item_id]
            for history_item_ids, new_item_id in zip(
                history_item_ids_list, new_item_id_list
            )
        ]

        return sequences
    
    def __add__(self, another_idrec_dataset: 'IDRecDataset') -> 'IDRecDataset':
        """
        Concat the current dataset with another.

        Parameters:
            another_idrec_dataset (IDRecDataset): The another dataset.
        
        Returns:
            IDRecDataset: The concatenated dataset.
        """

        assert (
            hasattr(another_idrec_dataset, "raw_data") and
            isinstance(another_idrec_dataset.raw_data, List)
        )
        
        self.raw_data += another_idrec_dataset.raw_data
        return self

    def __getitem__(self, index):
        sequence = self.raw_data[index]
        history_item_ids = sequence[:-1]
        new_item_id = sequence[-1]

        history_item_ids += [0] * max(0, self.max_len - len(history_item_ids))

        return {
            "item_seqs": torch.tensor(history_item_ids, dtype=torch.long),
            "labels": torch.tensor(new_item_id, dtype=torch.long),
            "seq_lengths": len(history_item_ids)
        }


class IDRecDatasets:
    def __init__(self, categories: List[str]):
        self.categories = categories
        self.train_dataset, self.valid_dataset, self.test_dataset = self.load_data_allcat()
    
    def load_data_allcat(self):
        train_dataset = None
        valid_dataset = None
        test_dataset = None
        for category in self.categories:
            if train_dataset is None:
                train_dataset = IDRecDataset(category=category, max_len=10, usage="train")
            else:
                train_dataset += IDRecDataset(category=category, max_len=10, usage="train")

            if valid_dataset is None:
                valid_dataset = IDRecDataset(category=category, max_len=10, usage="valid")
            else:
                valid_dataset += IDRecDataset(category=category, max_len=10, usage="valid")

            if test_dataset is None:
                test_dataset = IDRecDataset(category=category, max_len=10, usage="test")
            else:
                test_dataset += IDRecDataset(category=category, max_len=10, usage="test")
    
        return train_dataset, valid_dataset, test_dataset