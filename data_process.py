import os
import json
from pathlib import Path
from typing import List, DefaultDict, Callable, Optional, TypeVar, Generic
from abc import ABC, abstractmethod

# Data processing
import pandas as pd
import csv

from tqdm import tqdm
from loguru import logger


logger.add("logs/data_process.log", rotation="10 MB")
T = TypeVar('T')

"""
Data Processing Infrastructures
"""

class CategoryLoader(ABC, Generic[T]):
    def __init__(self, category: str, ext: str, content: str):
        self.ext = ext
        self.category = category
        self.content = content

    @abstractmethod
    def _load(self, file_path, func: Callable = lambda x: x) -> T:
        pass
    
    def __call__(self, func: Callable = lambda x: x) -> T:

        file_name = f"{self.category}.{self.ext}"

        if self.content != '':
            file_name = f"{self.content}_{file_name}"
        
        file_path = os.path.join(
            "data", "raw", self.category, file_name)

        if not file_path.endswith(f".{self.ext}"):
            message = (
                f"[CSIT5210 Err]: The file path you inputted " 
                f"is not a valid .{self.ext} file. ({file_path})")
            logger.error(message)
            raise ValueError(message)
        
        if not os.path.exists(file_path):
            message = (
                f"[CSIT5210 Err]: The file"
                f" {file_path} does not exist.")
            logger.error(message)
            raise FileNotFoundError(message)
    
        return self._load(file_path, func)


class JsonlLoader(CategoryLoader[List[dict]]):
    def __init__(self, category: str, content: str = ""):
        super().__init__(
            category=category, 
            ext="jsonl", 
            content=content)
    
    def _load(self, file_path, func: Optional[Callable]) -> List[dict]:
        dict_lines = []
        with open(file_path, "r") as file_lines:
            for i, line in tqdm(enumerate(file_lines)):
                dict_lines.append(func(json.loads(line)))
            file_lines.close()
        return dict_lines


class CSVLoader(CategoryLoader[pd.DataFrame]):
    def __init__(self, category: str, content: str = ""):
        super().__init__(
            category=category, 
            ext="csv", 
            content=content)
    
    def _load(self, file_path, func: Optional[Callable]) -> pd.DataFrame:
        df = pd.read_csv(file_path, sep=",")
        return df

"""
Data Graining
"""


def load_raw_data(category: str):
    """
    Load key contents of review and meta.
    Parameters:
        category (str): 
            The category of the dataset.
            E.g., "Video_Games", "Books", etc.
    Returns:
        df_user_interact (np.DataFrame): 
            A list of user interactions:
            Which user interacted with which
            item at which time.
        parentasin_title_map (List[dict]): 
            A map from parent asin to title.
    """
    _parentasin_title_map = JsonlLoader(category=category, content='meta')(
        func=lambda record:{
        "parent_asin": record["parent_asin"], 
        "title": record["title"]})
    
    parentasin_title_map = {
        record['parent_asin']: record['title'] 
        for record in _parentasin_title_map}
    
    title_itemid_map = {
        title: i for i, title in enumerate(parentasin_title_map.values())}
    
    df_user_interact = CSVLoader(category=category, content='')()

    return df_user_interact, parentasin_title_map, title_itemid_map

# def global_scan():
#     for category in os.listdir('data/raw'):   ``
#         file = os.path.join('data/raw', category, f'{category}.csv')
#         df = pd.read_csv(file, sep=',')
#         num_unique = len(df['parent_asin'].unique())
#         print(f"{category} - {num_unique}")


def get_5core_ui_list(
        df_user_interact: pd.DataFrame, 
        parentasin_title_map: pd.DataFrame,
        title_itemid_map: dict) -> pd.DataFrame:
    
    # Sort the list by user_id, then by timestamp
    # To build up leave-one-out dataset
    df_user_interact.sort_values(['user_id', 'timestamp'])

    # Search item title from paren_asin
    df_user_interact['item_title'] = df_user_interact['parent_asin'].map(parentasin_title_map)

    # Use sequencial item ID over strings
    df_user_interact['item_id'] = df_user_interact['item_title'].map(title_itemid_map)

    # List out key columns of each user
    key_concerns = ['parent_asin', 'timestamp', 'item_title', 'item_id']

    # Group users to list the above columns
    user_group = df_user_interact.groupby('user_id')
    user_data = user_group.agg({
        key_concern: list 
        for key_concern in key_concerns
    }).to_dict('index')

    train_list: List[dict] = []
    valid_list: List[dict] = []
    test_list: List[dict] = []
    for user_id, interaction in user_data.items():

        (parentasin_list,
         timestamp_list,
         itemtitle_list,
         itemid_list) = [
             interaction[key_concern] 
             for key_concern in key_concerns]
        
        interaction_length = len(parentasin_list)

        for ptr_seq_end in range(1, interaction_length):
            new_record = {
                'user_id': user_id,
                'history_item_asins': parentasin_list[:ptr_seq_end][-10:],
                'new_item_asin': parentasin_list[ptr_seq_end],
                'history_item_titles': itemtitle_list[:ptr_seq_end][-10:],
                'new_item_title': itemtitle_list[ptr_seq_end],
                'history_item_ids': itemid_list[:ptr_seq_end][-10:],
                'new_item_id': itemid_list[ptr_seq_end],
                'new_item_timestamp': timestamp_list[ptr_seq_end]
            }

            # Mostly, new records are given to training set.
            # For the last two records for each user, it will be given 
            # to the validation set and test set respectively.

            if ptr_seq_end < interaction_length - 2:
                train_list.append(new_record)
            elif ptr_seq_end == interaction_length - 2:
                valid_list.append(new_record)
            elif ptr_seq_end == interaction_length -1:
                test_list.append(new_record)
            else:
                message = (f"[CSIT5210 Error]: "
                           f"Failed to construct leave-one-out dataset."
                           f"Invalid pointer value {ptr_seq_end}.")
                logger.error(message)
                raise ValueError(message)
    
    df_train = pd.DataFrame(train_list)
    df_valid = pd.DataFrame(valid_list)
    df_test = pd.DataFrame(test_list)
    
    return df_train, df_valid, df_test
    

def build_asin_itemid_map(ui_df: pd.DataFrame):
    """
    Obtaining a pruned user interaction list,
    assign each item parent_asin a unique
    numerical item id, and organize a hashmap.

    Parameters:
        ui_df (pd.DataFrame): 
            A dataframe of user interactions.
    Returns:
        asin_to_itemid (dict): 
            A dictionary mapping parent_asin to item_id.
    """
    
    unique_asins = sorted(ui_df['parent_asin'].unique())
    asin_to_itemid = {asin: i for i, asin in enumerate(unique_asins)}

    return asin_to_itemid


# def grain_user_interaction_list(
#         ui_df: pd.DataFrame, 
#         asin_to_itemid: dict):
#     ui_df["item_id"] = ui_df["parent_asin"].map(asin_to_itemid)
#     ui_df = ui_df.sort_values(["user_id", "timestamp"])

#     user_itemlist = ui_df.groupby("user_id")["item_id"]


if __name__ == "__main__":
    # global_scan()

    df_ui, parentasin_title_map, title_id_map = load_raw_data(category='Video_Games')

    df_train, df_valid, df_test = get_5core_ui_list(
        df_ui, parentasin_title_map, title_id_map)

    df_train.to_csv('data/grained/Video_Games/train_Video_Games.csv', index=False)
    df_valid.to_csv('data/grained/Video_Games/valid_Video_Games.csv', index=False)
    df_test.to_csv('data/grained/Video_Games/test_Video_Games.csv', index=False)

    
    # reviews, metas = load_raw_data(category="Video_Games")

    # df = get_pruned_user_interaction_list(reviews)
    # map = build_asin_itemid_map(df)

    # for k, v in map.items():
    #     print(f"{k} {v}")