import os
import json
from typing import List, Callable, TypeVar, Generic, Tuple
from abc import ABC, abstractmethod

# Data processing
import pandas as pd

from tqdm import tqdm
from loguru import logger


logger.add("logs/data_process.log", rotation="10 MB")
T = TypeVar('T')

"""
Data Processing Infrastructures
"""

class CategoryLoader(ABC, Generic[T]):
    """
    Loads dataset from a specific category.
    """
    
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
    """
    Loads raw dataset in .jsonl format from a 
    specific category.
    """
    def __init__(self, category: str, content: str = ""):
        super().__init__(
            category=category, 
            ext="jsonl", 
            content=content)
    
    def _load(self, file_path, func: Callable=lambda x: x) -> List[dict]:
        dict_lines = []
        with open(file_path, "r") as file_lines:
            for i, line in tqdm(enumerate(file_lines)):
                dict_lines.append(func(json.loads(line)))
            file_lines.close()
        return dict_lines


class CSVLoader(CategoryLoader[pd.DataFrame]):
    """
    Loads raw dataset in .csv format from a 
    specific category.
    """
    def __init__(self, category: str, content: str = ""):
        super().__init__(
            category=category, 
            ext="csv", 
            content=content)
    
    def _load(self, file_path, func: Callable=lambda x: x) -> pd.DataFrame:
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


def get_5core_ui_list(
        df_user_interact: pd.DataFrame, 
        parentasin_title_map: pd.DataFrame,
        title_itemid_map: dict) -> Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Using the raw dataset, build up a list of leave-one-out
    user interactions. The items are represented by:
        - parent_asin: The parent asin of the item.
        - item_title: The title of the item (further used).
        - item_id: The item ID.
    Parameters:
        df_user_interact (pd.DataFrame): 
            A list of user interactions:
            Which user interacted with which
            item at which time.
        parentasin_title_map (List[dict]): 
            A map from parent asin to title.
        title_itemid_map (dict): 
            A map from title to item ID.
    Returns:
        train_list (List[dict]): 
            A list of training records.
    """
    
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


def grain_dataset(categories: List[str]):
    """
    Grain a list of given categories in the dataset.
    Parameters:
        categories (List[str]): 
            A list of categories to be grained.
    """

    logger.info('[CSIT5210 Info]: \n\nGraining dataset started!\n\n')

    for category in categories:

        logger.info(
            f'[CSIT5210 Info]: \n\nGraining category {category} dataset...\n\n')
        base_path = f'data/grained/{category}'

        if not os.path.exists(base_path):
            logger.info(
                f'[CSIT5210 Info]: Base path {base_path} does not exist. '
                f'Creating...')
            os.makedirs(base_path)

        logger.info(
            f'[CSIT5210 Info]: \n\nLoading {category} raw 5-core data...\n\n')
        df_ui, pa_title_map, title_id_map = load_raw_data(category=category)

        logger.info(
            f'[CSIT5210 Info]: \n\nGraining {category} data...\n\n')
        df_train, df_valid, df_test = get_5core_ui_list(
            df_user_interact=df_ui, 
            parentasin_title_map=pa_title_map, 
            title_itemid_map=title_id_map)

        logger.info(
            f'[CSIT5210 Info]: \n\nSaving {category} train, '
            f'valid and test .csv files...\n\n')
        
        df_train.to_csv(os.path.join(base_path, f'train_{category}.csv'), index=False)
        df_valid.to_csv(os.path.join(base_path, f'valid_{category}.csv'), index=False)
        df_test.to_csv(os.path.join(base_path, f'test_{category}.csv'), index=False)


if __name__ == "__main__":
    grain_dataset(categories=[
        'Arts_Crafts_and_Sewing',
        'Baby_Products',
        'Movies_and_TV',
        'Sports_and_Outdoors',
        'Video_Games'
    ])
