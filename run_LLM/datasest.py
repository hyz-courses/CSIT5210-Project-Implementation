import os
import ast

from typing import cast, List, Optional, Tuple

import numpy as np
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
        self.raw_data, self.max_item_id = self.load_data()
    
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

        # Merge history item id list and new item id
        # into a single list.
        sequences = [
            cast(List, ast.literal_eval(history_item_ids)) + [new_item_id]
            for history_item_ids, new_item_id in zip(
                history_item_ids_list, new_item_id_list
            )
        ]

        max_item_id = np.max(sequences)

        return sequences, max_item_id
    
    def __add__(self, another_idrec_dataset: Optional['IDRecDataset']) -> 'IDRecDataset':
        """
        Concat the current dataset with another.

        Parameters:
            another_idrec_dataset (IDRecDataset): The another dataset.
        
        Returns:
            IDRecDataset: The concatenated dataset.
        """

        if another_idrec_dataset is None:
            return self

        assert (
            hasattr(another_idrec_dataset, "raw_data") and
            isinstance(another_idrec_dataset.raw_data, List)
        )
        
        self.raw_data += another_idrec_dataset.raw_data
        self.max_item_id = max(
            self.max_item_id, 
            another_idrec_dataset.max_item_id)
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
    """
    An train, valid and test item-id dataset
    for a specific category.
    """
    
    def __init__(self, category: str):
        self.category = category
    
    def get_datasets(self) -> Tuple[
        IDRecDataset, IDRecDataset, IDRecDataset,
        int, List[int]
    ]:
        """
        Load the train, valid and test ID datasets
        for a specific category, along with some stats.
        """

        train_dataset = IDRecDataset(category=self.category, max_len=10, usage="train")
        valid_dataset = IDRecDataset(category=self.category, max_len=10, usage="valid")
        test_dataset = IDRecDataset(category=self.category, max_len=10, usage="test")

        whole_dataset = train_dataset + valid_dataset + test_dataset
        total_item_num = whole_dataset.max_item_id
        select_pool = [1, total_item_num + 1]

        return (train_dataset, valid_dataset, 
                test_dataset, total_item_num, select_pool)